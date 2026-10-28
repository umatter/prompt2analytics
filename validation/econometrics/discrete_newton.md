# Validation: Newton loops of multinomial and ordered logit/probit (RF-31, RF-32)

Registry: [`validation/methods/discrete_newton.json`](../methods/discrete_newton.json)
(methods `multinom_newton_nnet` and `ordered_newton_many_categories`). R cases:
[`validation/r/cases/discrete_newton.R`](../r/cases/discrete_newton.R).

## Purpose

Two defects made `run_multinom` and `run_ordered_logit` / `run_ordered_probit` return
wrong fits, some of them labelled converged.

- **RF-31.** When Cholesky failed, `linalg::matrix_ops::safe_inverse` fell back to faer
  0.22.6's full-pivot LU; `matrix_inverse` and `solve` used it for every input. Nothing
  checked the result. That LU returns wrong or all-NaN inverses for well-conditioned
  matrices of size 16 or more with sparse structure: diagonal and tridiagonal matrices,
  the Kronecker-structured Hessian of the multinomial logit, the ordered-model Hessian,
  `I - rho W` for spatial weights. Dense random matrices of the same sizes, and every
  matrix below size 16, came out right. The Newton loops of both estimators invert a
  negative definite Hessian, which always takes that fallback, and their convergence test
  `max_change.max(change.abs())` drops NaN, so an all-NaN step counted as converged. An
  8-category multinomial fit (35 parameters) returned coefficients of -2e11 and a
  log-likelihood of -inf; ordered fits with 16 or more categories returned NaN for every
  estimate, a log-likelihood of `n ln(1e-15)`, and `Converged: true`.
- **RF-32.** `run_multinom` computed probabilities as `exp(x'b).min(1e10) / sum`. Once a
  linear predictor exceeds ln(1e10) = 23.03, as happens for a well-predicted category,
  the clamp biases the score and the likelihood: on five separated categories the fit
  stopped at a log-likelihood of -21.6960, above the true maximum -22.6349, labelled
  converged.

## The verified fallback

`safe_inverse` keeps its Cholesky branch bit for bit, so every symmetric positive definite
caller (X'X, information matrices, covariance matrices) is unchanged. When Cholesky fails,
and in `matrix_inverse` always:

1. a non-finite entry is refused with `LinalgError::NonFinite`;
2. `cond = condition_number(m)`; above `CONDITION_THRESHOLD` (1e10) `safe_inverse`
   returns the pseudoinverse with the warning `Some(cond)` and `matrix_inverse` returns
   `IllConditioned`, as before;
3. otherwise the inverse by LU with partial pivoting is accepted only if it is finite and
   `max |A X - I| <= INVERSE_RESIDUAL_FACTOR * n * eps * cond`, with
   `INVERSE_RESIDUAL_FACTOR = 100`; if it is not, the SVD pseudoinverse under the same
   check; if neither passes, `SingularMatrix`.

`solve` checks the normwise backward error instead:
`||A x - b|| <= 100 n eps (||A|| ||x|| + ||b||)` (infinity norms), with the SVD route
under the same check for `cond <= 1e10` and `SingularMatrix` otherwise (before: a singular
`A` gave `Ok([-inf, inf])`).

Why 100: in a planning sweep over 435 matrices with `cond <= 1e10` (random dense, negative
definite, symmetric indefinite, sparse structured, graded, and multinomial Hessians up to
495 x 495), correct inverses scored at most 0.37 (partial pivoting) and 0.75 (SVD) in
units of `n * eps * cond`, and every wrong full-pivot result scored at least 5.4e12 or
was NaN. LAPACK's own inverse test (`dget03`) accepts ratios up to 30. The bound uses
`cond` from the SVD, not the candidate's norm, so a wrong candidate cannot widen it. A
mutation run showed that the gate alone stops wrong results: with the full-pivot LU put
back as the candidate, every residual-based test still passes through the SVD route.

## R cases

`Rscript validation/r/run.R --write disc_multinom_mn8 disc_multinom_sep5 disc_multinom_j4
disc_polr_j5 disc_polr_j16 disc_polr_j30` writes the six JSON files and the six generated
datasets; `Rscript validation/r/run.R --check` recomputes them and fails on any
difference. Computed with R 4.3.3, nnet 7.3.19, MASS 7.3.60.0.1.

| Case | Dataset | n | Categories | Parameters | Purpose |
|---|---|---|---|---|---|
| `disc_multinom_mn8` | `generated/multinom_mn8.csv` | 500 | 8 | 35 | RF-31: the starting Hessian is well conditioned (kappa 583.6) but not positive definite |
| `disc_multinom_sep5` | `generated/multinom_sep5.csv` | 30 | 5 | 8 | RF-32: largest x'b at the MLE 34.76 > ln(1e10) |
| `disc_multinom_j4` | `generated/multinom_j4.csv` | 400 | 4 | 9 | control (largest x'b 3.33) |
| `disc_polr_j5` | `generated/ordered_j5.csv` | 30 | 5 | 5 | control below the size where the old LU failed |
| `disc_polr_j16` | `generated/ordered_j16.csv` | 96 | 16 | 16 | RF-31 |
| `disc_polr_j30` | `generated/ordered_j30.csv` | 180 | 30 | 30 | RF-31 |

The multinomial references (`coef_tight`, `se_tight`, `z_tight`, `p_tight`) are nnet's
estimate re-solved by Newton's method on the analytic score and information until the
relative step is below 1e-15; nnet's own `vcov` at `reltol = 1e-16` agrees with them to
1e-6 (asserted in R). Labels sort as strings (`c001`, `c002`, ...), which is the order
p2a uses.

## Multinomial logit

R values (nnet, re-solved):

| Case | Row | (Intercept) | x1 / x | x2 | log-likelihood |
|---|---|---|---|---|---|
| mn8 | `2` coef | 0.3621139591700734 | 1.1997682999565642 | 0.8867212910396424 | -790.4903598151795 |
| mn8 | `2` SE | 0.5075995347899822 | 0.2594004401525811 | 0.26515143556342285 | |
| sep5 | `c004` coef | -22.24002804665748 | 8.28945693334661 | | -22.63486081368176 |
| sep5 | `c005` coef | -25.745747121894112 | 9.036425590004757 | | |
| sep5 | `c005` SE | 9.545850925987375 | 3.110067730574075 | | |
| j4 | `b` coef | 0.3214539512039905 | 1.0380840901708501 | -0.6307326759172206 | -469.05692817200196 |
| j4 | `b` SE | 0.16042640380822126 | 0.17959963261079093 | 0.1749596287707518 | |

Measured agreement of `run_multinom` (largest relative difference over every reported
number; all three fits report `converged = true`):

| Case | Iterations | Coefficients | SE | z | log-likelihood, AIC, BIC |
|---|---|---|---|---|---|
| mn8 | 7 | 7.3e-13 (absolute 6.8e-15) | 3.9e-15 | 7.3e-13 | 1.2e-15 |
| sep5 | 10 | 8.0e-16 | 1.5e-13 | 1.5e-13 | 7.9e-16 |
| j4 | 6 | 6.3e-15 | 1.7e-16 | 6.1e-15 | 3.4e-16 |

p-values agree to 5e-9 relative and 5e-11 absolute; they are computed as
`2 * (1 - Phi(|z|))`, which loses relative precision in the tail (X-17). The start
Hessian of mn8 matches R's `kappa(exact = TRUE)` and largest eigenvalue to 4.0e-14 and
3.6e-14.

Before the fix: mn8 `2:(Intercept)` -215218243657.17, log-likelihood -inf, AIC and BIC
inf, not converged after 50 iterations; the full-pivot inverse of the starting Hessian had
`max |H X - I|` = 154.6. sep5 stopped at the clamped optimum: log-likelihood
-21.696035498161688, `c005` (-28.33345127945314, 9.66743522531738). j4 passed before as
well.

The Newton step now solves the information matrix (positive definite at any interior
point, so it takes the Cholesky path), and probabilities, the log-likelihood and the final
information use log-sum-exp with `m = max(0, max_j x'b_j)`.

**Convergence contract.** `converged == true` implies that every coefficient and the
log-likelihood are finite and that the last step came from a regular (not pseudo-)
inverse with largest absolute change below 1e-8. A non-finite step, or a step from the
pseudoinverse (singular information), ends the iteration with `converged = false` and is
not applied. With 100 categories of one row each (complete separation, no MLE) the fit
stops at its first pseudoinverse step, iteration 35, with finite estimates; without that
rule it reported `converged = true` with a log-likelihood of -5e16. That fit takes 32.7 s
in an opt-level 0 build (CI's profile), so the tests check the same behaviour on 20
categories of one row each (38 parameters, 0.5 s): the fit stops at iteration 42, and
without the rule it reports `converged = true` at iteration 45 with a log-likelihood of
-1.5e18. The 100-category test checks only that the category cap admits 100 categories
and refuses 101.

Tolerances (`validation/README.md`): n = 400-500 coefficients 1e-8, SE and statistics
1e-6; n = 30 coefficients 1e-6, SE and statistics 1e-5 (relative); p-values also 1e-12
absolute; counts exact.

## Ordered logit and probit

MASS::polr references (`Hess = TRUE, control = list(reltol = 1e-14)`, convergence 0 for
every fit; every category has 6 rows):

| Case | n | null log-likelihood | logit x (SE) | logit logLik | probit x (SE) | probit logLik |
|---|---|---|---|---|---|---|
| j5 | 30 | -48.28313737302301 | 1.7754875148526819 (0.39944339649894356) | -29.069992276590806 | 1.033659829959397 (0.20960479140385097) | -29.12981222882675 |
| j16 | 96 | -266.168517335019 | 1.3770288290149235 (0.1647102655189493) | -219.81083320302423 | 0.7926197338497165 (0.08724936652093264) | -220.30804206346016 |
| j30 | 180 | -612.215528699188 | 1.3006515849957276 (0.11418223227312636) | -527.9631621310365 | 0.7636542338575049 (0.06127658157121911) | -526.9462777364893 |

The case files also carry every threshold (`logit_zeta`, `probit_zeta`, named
`c001|c002`, ...) with its SE, and polr's own null fits.

What is pinned for both links: n and the category counts (exact), the null log-likelihood
(`Tol::stat(n)`: 1e-5 relative for n = 30 and 96, 1e-6 for n = 180), finite coefficients,
thresholds and log-likelihood, a log-likelihood not above polr's maximum, and `converged`
only above the null model.

**Deviations** (owned by T2-WP10): ECON-4, the beta score has the wrong sign and the
outer-product Hessian divides by `p_i` twice, so the fit never reaches polr's MLE; ECON-5,
the standard errors are a 0.1 placeholder. After this package the fits are finite and
`converged = false` after 100 iterations (j16: log-likelihood -376.4779 logit, -356.0706
probit; j30: -808.2087, -753.3586); before it, j16 and j30 were all NaN with a
log-likelihood of -3315.7225 and -6216.9798 and `Converged: true` after 1 to 3
iterations. The polr values are polr's own optimum (BFGS at `reltol = 1e-14`), not
re-solved, so T2-WP10 either pins them at `Tol::iterative` with the measured need or adds
tight values to its own case file.

The ordered Newton step, like the multinomial one, solves the information matrix: the
negated outer-product Hessian plus the 1e-6 ridge, which is positive definite and so takes
the Cholesky path at any conditioning. The step used to invert the negative definite
Hessian, which always fails Cholesky; `safe_inverse` then returns the pseudoinverse
whenever `cond > 1e10`, which means ill-conditioning, not singularity, and a regressor in
the thousands is enough for it. Under the pseudoinverse-step rule below, `disc_polr_j5`
with x multiplied by 1e4 then returned its starting point after one iteration
(coefficient 0, the null log-likelihood). On the information matrix the fit with x
multiplied by 1e4 or 1e6 is the unscaled one with the coefficient divided by the scale,
to 1.1e-10 relative (logit) and 7.9e-11 (probit); the difference comes from the ridge,
which is not scale-free. The j16 and j30 log-likelihoods above are unchanged to the digits
shown.

The ordered loop has the same contract as the multinomial one: a non-finite score or
step, or a step from the pseudoinverse, ends the iteration unapplied and not converged.
On the information matrix a pseudoinverse step occurs only when Cholesky fails in floating
point despite the ridge and the condition number exceeds 1e10. The 100-category fit of one
row each (no MLE) stops on such a step, at iteration 18; with pseudoinverse steps applied
it runs all 100 iterations.

## Known differences

- `run_multinom`'s final covariance falls back to the identity matrix if the information
  cannot be inverted (T3-WP18).
- No rank or aliasing check of the design (ECON-61, T3-WP18).
- Rows with a null outcome (ECON-12, T2-WP18).
- `p2a discrete mlogit` prints no convergence flag (CLI-17, T2-WP18).
- p-values below about 1e-16 print as 0 (X-17, Tier 3).
- The ordered category order comes from sorting the labels as strings (ECON-19,
  T2-WP10); these cases avoid it with `c001`-style labels.

## Tests

R-pinned (load a case):

- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_mn8`
- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_start_hessian_mn8`
- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_sep5`
- `econometrics::discrete::multinomial_logit::tests::test_validate_multinom_j4`
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_j5`
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_j16`
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_j30`
- `econometrics::discrete::ordered_models::tests::test_validate_ordered_j5_regressor_scale` (loads `disc_polr_j5`'s dataset and checks the fit is unchanged when x is multiplied by 1e4 or 1e6)

Helpers, the category cap and the separated fits:

- `econometrics::discrete::multinomial_logit::tests::test_softmax_into_log_sum_exp`
- `econometrics::discrete::multinomial_logit::tests::test_max_abs_step_rejects_non_finite`
- `econometrics::discrete::multinomial_logit::tests::test_multinom_accepts_100_categories`
- `econometrics::discrete::multinomial_logit::tests::test_multinom_one_row_per_category_does_not_converge`
- `econometrics::discrete::ordered_models::tests::test_ordered_max_abs_step_rejects_non_finite`
- `econometrics::discrete::ordered_models::tests::test_ordered_accepts_100_categories`

Linear algebra (`linalg::matrix_ops::tests::`; R `solve()` values for `-tridiag(-1, 3, -1)`
and `I - 0.5 W` on a 5 x 5 rook grid):

- `test_safe_inverse_lu_fallback_negative_diagonal`
- `test_safe_inverse_lu_fallback_tridiagonal`
- `test_safe_inverse_lu_fallback_sparse_sweep`
- `test_matrix_inverse_structured`
- `test_solve_structured_and_singular`
- `test_linalg_rejects_non_finite`
- `test_safe_inverse_spd_keeps_cholesky_path`
- `test_safe_inverse_ill_conditioned_keeps_pseudoinverse`
- `test_verified_inverse_rejects_bad_candidates`

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- linalg::matrix_ops:: econometrics::discrete::
Rscript validation/r/run.R --check
```
