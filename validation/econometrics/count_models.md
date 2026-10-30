# Validation: Count Models (MASS::glm.nb, pscl::zeroinfl)

Registry: [`validation/methods/discrete_ordered_count.json`](../methods/discrete_ordered_count.json)
(methods `discrete_negbin`, `discrete_zip`, `discrete_zinb`). R cases: `disc_negbin` and
`disc_zeroinfl` in [`validation/r/cases/discrete.R`](../r/cases/discrete.R). The hurdle
models are in [hurdle.md](hurdle.md).

## Method Overview

### Negative binomial

`run_negbin(dataset, y, x, init_theta)` fits the NB2 model of `MASS::glm.nb`:

    y ~ NB(mu, theta),   log(mu) = x'beta,   Var(y) = mu + mu^2 / theta

`init_theta` is the starting value of theta (glm.nb's `init.theta`); it must lie in
`[1e-8, 1e6]` (`InvalidSpecification` naming `init_theta` and both bounds otherwise).
Below the range the theta score and information cannot be evaluated (trigamma(theta) =
1/theta^2 overflows below 1e-154, so theta could not move); above it the theta score is
at rounding level (near 1e8), so the alternation stopped at its start with the
no-overdispersion warning, also for overdispersed data.

Algorithm (glm.nb's alternation):

- Start: the coefficients of a Poisson fit (glm's IRLS from `mu = y + 0.1`), or of the NB
  fit at `init_theta` when given; theta from `init_theta` or MASS's moment start
  `n / sum((y / mu - 1)^2)`, moved into `[1e-8, 1e6]`.
- Coefficients: Fisher scoring (IRLS) at fixed theta, weights `W = mu theta / (theta + mu)`,
  working response `z = eta + (y - mu) / mu`, at most 100 iterations, until
  `max |delta beta| < 1e-12 (1 + max |beta|)` or the log-likelihood changes by at most
  `1e-12 (|ll| + 0.1)` (glm.fit's relative test, on the log-likelihood). The second test
  is the one an ill-conditioned `X'WX` passes (polynomial terms, nearly collinear
  regressors): there the solve moves the coefficients by rounding noise in the directions
  the data determine poorly, which does not change the likelihood. A design whose `X'WX`
  is not positive definite is `SingularMatrix`.
- No finite MLE: an outcome without a finite MLE (an all-zero outcome, or zero counts a
  regressor separates) is recognised by the zero counts the final IRLS is still driving to
  0. The working response of such a row is `eta - 1`, so each step lowers its linear
  predictor by about 1, at any n (measured: 1.0 on the all-zero and the separated
  fixtures). When the last step of the final IRLS lowered the linear predictor of a zero
  count by more than 0.01 (its fitted mean by more than 1%), the fit gives
  `converged = false` and a warning that counts those rows. The fitted means are not the
  test: the IRLS stops on its log-likelihood test, whose threshold grows with `|ll|`, so on
  larger data it stops while they are still above glm.fit's `10 eps` (5000 rows with a
  dummy on every 50th row, whose rows all have y = 0). An earlier verdict that needed such
  means reported that fit as converged, without a warning. At a regular fit the last step
  moves every row by rounding, also a zero count at an extreme regressor value, which adds
  nothing to the likelihood. Fitted means below `10 eps` (glm.fit's "fitted rates
  numerically 0" test) are then a note in `warnings` that counts the rows ("fitted means
  numerically 0 occurred for k rows ... at converged coefficients"), with `converged`
  unchanged (`disc_negbin` plus the row `y = 0, x1 = -80, x2 = 0.1`: the last step moves
  that row's linear predictor by 5e-11, and the fit is the 400-row one; R's glm.nb
  converges there with theta 1.602585 and glm.fit's warning from its Poisson start). A
  test on the coefficient change alone cannot tell the two apart: theta's last change
  moves the coefficients of a regular fit by more than `1e-12 (1 + max |beta|)` too. A
  final IRLS that settles on neither test also gives `converged = false`, with its own
  warning.
- theta: Newton steps on `ln theta` at fixed `mu` with MASS `theta.ml`'s score
  `s = sum(digamma(theta + y) - digamma(theta) + ln theta + 1 - ln(theta + mu) - (y + theta) / (mu + theta))`
  and information
  `I = sum(-trigamma(theta + y) + trigamma(theta) - 1/theta + 2/(mu + theta) - (y + theta)/(mu + theta)^2)`
  (MASS's signs), clamped to +-2 and halved while the profile log-likelihood falls. A
  theta at which the score, the information or the log-likelihood is not finite ends the
  alternation with `converged = false` and a warning that names theta.
- The two alternate (at most 100 rounds) until theta changes by less than 1e-10 relative
  and the log-likelihood by less than 1e-12 relative; then one more IRLS at the final
  theta.
- Standard errors: `(X'WX)^-1` at the estimate (summary.negbin's expected information);
  `SE(theta) = 1 / sqrt(I)` at the optimum (MASS `SE.theta`). `z = b / SE`,
  `p = 2 P(Z > |z|)` from the upper tail. AIC and BIC count `k + 1` parameters.
- `trigamma(x)`: the recurrence `psi1(x) = psi1(x + 1) + 1/x^2` up to `x >= 10`, then the
  asymptotic series `1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7) - 1/(30x^9) +
  5/(66x^11)`, accurate to about 2e-13 relative (R's `trigamma` at the points the case
  lists).

theta is bounded by 1e8. Without overdispersion the likelihood keeps rising towards the
Poisson limit: a final theta above 1e6 (the NB2 variance is then within 1e-6 mu^2 of the
Poisson one) gives `converged = false`, `theta_std_error = NaN` and a warning that the
data show no overdispersion. R's glm.nb stops there with "iteration limit reached" (on the
12-row unit-test fixture: theta 106936.84).

### Zero-inflated Poisson and NB

`run_zip(dataset, y, x, z)` and `run_zinb(...)` fit `pscl::zeroinfl(y ~ x | z, dist =
"poisson" | "negbin")` with a logit zero part (`z` defaults to `x`, pscl's default):

    P(y = 0) = pi + (1 - pi) f(0),   P(y = k) = (1 - pi) f(k) for k > 0,
    logit(pi) = z'gamma,   log(mu) = x'beta,   f Poisson(mu) or NB2(mu, theta)

Algorithm (pscl's `EM = TRUE` start, then a full maximisation):

- Start: `gamma` from a logistic IRLS of `1[y = 0]` on `Z`, `beta` from a Poisson IRLS of
  `y` on `X` (glm.fit's guards: the logit mean within `[eps, 1 - eps]`, `d mu / d eta >=
  eps`, `eta <= 700`; a singular or non-finite update keeps the previous coefficients).
- ZIP EM to convergence (relative log-likelihood change below 1e-8, at most 500 rounds):
  the E-step `w_i = pi_i / (pi_i + (1 - pi_i) e^-mu_i)` for zeros, the M-steps a logistic
  IRLS with response `w` and a Poisson IRLS with weights `1 - w`; a round that would lower
  the log-likelihood is not taken. ZINB starts from the ZIP EM values and `ln theta = 0`
  (pscl's start when `EM = FALSE`; the optimum is the same).
- Joint maximisation of the log-likelihood in `(beta, gamma[, ln theta])`, computed in log
  space, by damped Newton steps with pscl's analytic gradient (`gradPoisson`,
  `gradNegBin`) and the analytic Hessian (formulas in the doc comment of `zi_loglik`,
  checked against central differences of the gradient); Levenberg damping and step
  halving as for the ordered models; `ln theta <= ln 1e8`; at most 200 iterations.
  Converged: the Newton decrement `g'(-H)^-1 g` (from the step just computed) at most
  `1e-12 (1 + |ll|)`, log-likelihood change below `1e-12 (|ll| + 1e-12)` and every
  Newton-step component at most `1e-10 (1 + |par_k|)`. A non-finite gradient or Hessian,
  a failed line search or the iteration limit end the fit with `converged = false` and a
  warning (which reports `max |grad|`). Newton's method with the exact Hessian is
  affine-invariant, and so is the decrement: with a regressor multiplied by 1e4, 1e6, 1e10
  or 1e12 the fit converges, its coefficients and SEs are the unscaled ones divided by the
  scale, and z and the log-likelihood are unchanged. (The earlier test
  `max |grad| < 1e-8 (1 + |ll|)` did not have this property: the gradient component of a
  regressor scales with its units, so ZIP with x1 times 1e10 reached the unscaled
  estimates but was reported as not converged after 200 iterations.)
  (The central-difference Hessian this replaced used a step relative to the parameter
  value, so its SEs moved with the units of a regressor: z by 10% at a scale of 1e5.)
- Standard errors from the inverse of the negated Hessian at the estimate (NaN with a
  warning if it is not positive definite); for ZINB `log_theta_std_error` = SE(ln theta)
  (pscl's `SE.logtheta`) and `theta_std_error = theta SE(ln theta)` (delta method). z and
  p as for NB; AIC and BIC count `kx + kz (+ 1)` parameters.
- Fitted zero-inflation probabilities with `min(pi, 1 - pi) < 10 eps` give the warning
  "fitted zero-inflation probabilities numerically 0 or 1 occurred for k rows (possible
  quasi-separation in the zero part, or rows with extreme regressor values); check those
  rows" (glm.fit's diagnostic; pscl's EM suppresses it and reports such fits as
  converged). It leaves `converged` unchanged: one row with an extreme regressor value is
  enough to trigger it on a regular fit. A ZINB theta above 1e6 is flagged as
  for NB, with a warning that names the zero-inflated Poisson model (pscl::zeroinfl with
  `dist = "poisson"`). Its ln theta direction of the Hessian is then singular, so the count
  and zero-part SEs come from their own block, the Poisson-limit information, and the theta
  SEs are NaN.
- An outcome without a zero count is refused (`InvalidSpecification`, as pscl's "minimum
  count is not zero"). Zeros are the values below 0.5 (ECON-58).

### Missing values

The outcome and the regressors are read together; rows with a missing value in any of
them are dropped (R `na.omit`) and reported in `missing` (JSON key, omitted when nothing
was dropped) and on the last line of the text output. A text column is
`NonNumericColumn`. Non-convergence and singular information are reported in `warnings`
(the `Warnings:` block of the text output).

## Reference Implementations

| Package | Version | Function |
|---|---|---|
| MASS | 7.3.60.0.1 (R 4.3.3) | `glm.nb(control = glm.control(epsilon = 1e-14, maxit = 200))`, `theta.ml` |
| base R | 4.3.3 | `trigamma` |
| pscl | 1.5.9 | `zeroinfl(dist = "poisson" \| "negbin", control = zeroinfl.control(EM = TRUE, reltol = 1e-15, maxit = 100000))` |
| numDeriv | 2016.8.1.1 | `hessian`, `jacobian` (polishing and SEs) |

## R reference

Case `disc_negbin` (`validation/r/cases/discrete.R`; dataset
`validation/datasets/generated/negbin.csv`, 400 rows, seed 404): `y` NB2 with size 1.8 and
`mu = exp(0.4 + 0.5 x1 - 0.6 x2)` (Int64), `x1` normal, `x2` uniform, and `y_na`, `x1_na`
with 3 and 2 missing cells (5 incomplete rows). `Rscript validation/r/run.R --write
disc_negbin` writes it.

glm.nb's own fit stops one `theta.ml` step early: its `SE.theta` is evaluated at the theta
before the reported one and differs from `1/sqrt(information)` at the optimum by 1.7e-5
relative. The case therefore polishes glm.nb's fit by Newton steps on the joint
`(beta, theta)` score until `max |grad| < 1e-9` and recomputes the SEs there; glm.nb's
values are kept as `*_pkg`.

```r
m <- glm.nb(y ~ x1 + x2, data = d, control = glm.control(epsilon = 1e-14, maxit = 200))
par <- .polish(c(coef(m), m$theta), function(p) .nb_grad(p, X, y))        # joint (beta, theta) score
W <- mu * th / (th + mu); se <- sqrt(diag(solve(crossprod(X * sqrt(W)))))  # summary.negbin
se_theta <- 1 / sqrt(.nb_info(th, mu, y))                                  # theta.ml information at the optimum
```

| key | R value |
|---|---|
| `coef` | (Intercept) 0.2765157352028252, x1 0.4789005785872501, x2 -0.5202761398295949 |
| `se` | 0.12067279681163258, 0.06760627373095648, 0.21578173425256997 |
| `z` | 2.291450455353743, 7.083670673716847, -2.4111222464298936 |
| `p` | 0.021937374252251444, 1.4038530380270747e-12, 0.015903516903478965 |
| `theta`, `se_theta` | 1.6025851429909947, 0.32721579365464054 |
| `loglik`, `aic`, `bic` | -564.3012478589189, 1136.6024957178379, 1152.5683539062697 |
| `y_mean`, `y_var`, `n` | 1.1375, 2.2642543859649122, 400 |
| `theta_pkg`, `se_theta_pkg` (glm.nb) | 1.6025851427753957, 0.32721030566007775 |
| `theta_init50_pkg` (glm.nb, `init.theta = 50`) | 1.6025851427753954 |
| `na_coef` (395 rows) | 0.26944112534137227, 0.47402773297864337, -0.5051689565670879 |
| `na_se` | 0.12176776202503943, 0.06857685945875105, 0.2182275750886312 |
| `na_theta`, `na_se_theta` | 1.5624842841513744, 0.3171253193416621 |
| `na_loglik`, `na_n_used`, `na_n_dropped` | -557.9807627670712, 395, 5 (`y_na` 3, `x1_na` 2) |
| `int_coef`, `int_se` (`glm.nb(y ~ 1)`) | (Intercept) 0.12883287184296743, 0.067352401417311913 |
| `int_theta`, `int_se_theta`, `int_loglik` | 1.0690413506823526, 0.1786307915194861, -590.7891356865365 |
| `trigamma` at `trigamma_x` = 0.1, 0.5, 1, 1.6025851429909947, 3.7, 6, 10, 25.5, 1000 | 101.43329915079275, 4.934802200544679, 1.6449340668482264, 0.8566167623849812, 0.31003785767003833, 0.18132295573711527, 0.10516633568168572, 0.03999466964956293, 0.0010005001666666335 |

Case `disc_zeroinfl` (dataset `validation/datasets/generated/zeroinfl.csv`, 500 rows, seed
808): `y` zero-inflated Poisson and `y_nb` zero-inflated NB (size 2) with
`logit(pi) = -0.4 + 0.6 x1 + 0.2 x2`, `mu = exp(0.6 + 0.25 x1 - 0.3 x2)` (Int64), and `y_na`,
`x2_na` with 3 and 2 missing cells (5 incomplete rows). zeroinfl's BFGS stops at
`max |grad|` 7.5e-6 (ZIP) and 7.1e-6 (ZINB); its ZINB theta is 5.3e-7 relative from the MLE.
The case polishes pscl's fit with pscl's analytic gradient to `max |grad|` 9.8e-14 (ZIP)
and 8.1e-12 (ZINB) and takes the SEs from `numDeriv::hessian` there; pscl's values are kept
as `*_pkg`. `Rscript validation/r/run.R --write disc_zeroinfl` writes it.

| key | ZIP (`y`) | ZINB (`y_nb`) |
|---|---|---|
| `*_count` (Intercept), x1, x2 | 0.6259760625813262, 0.18647905374652335, -0.2665448461538206 | 0.41210175578854347, 0.2928331023453584, -0.34469422957446477 |
| `*_zero` | -0.35893642636304074, 0.4216190725590943, 0.1843533718931378 | -1.3270691586609378, 0.7774559970541303, 0.4259557829270744 |
| `*_se_count` | 0.05501336912265871, 0.05319705963870614, 0.06063009261132024 | 0.1260329943978712, 0.0819981352226419, 0.08343840402822274 |
| `*_se_zero` | 0.12617385646381543, 0.1352377225436285, 0.1351834256116356 | 0.5841342107431455, 0.3373393530304863, 0.2582986823793917 |
| `*_z_count` | 11.378617099157111, 3.505439116616914, -4.396246726234656 | 3.269792626584648, 3.5712166081613437, -4.131122036536955 |
| `*_p_count` | 5.344041039890042e-30, 0.0004558547486653013, 1.101387377357464e-05 | 0.0010762635342585198, 0.0003553268754454276, 3.609968617545867e-05 |
| `*_z_zero` | -2.844776536302334, 3.117614409863176, 1.3637276245888388 | -2.2718565943477573, 2.3046703269863373, 1.6490822911029268 |
| `*_p_zero` | 0.004444258969631584, 0.0018232116615723565, 0.17265330637629114 | 0.023095173294476534, 0.021185043961690894, 0.09913077708359233 |
| `*_loglik`, `*_aic`, `*_bic` | -688.7967373111242, 1389.5934746222483, 1414.8811232127814 | -712.3927772952783, 1438.7855545905566, 1468.287811279512 |
| `*_n`, `*_n_zeros` | 500, 259 | 500, 273 |
| `zinb_theta`, `zinb_log_theta` | | 1.1216047965897817, 0.11476051384912063 |
| `zinb_se_log_theta`, `zinb_se_theta` | | 0.3065579537099327, 0.3438368713138088 |
| `*_count_pkg` (pscl) | 0.6259760539704734, 0.18647906400263853, -0.26654483687991976 | 0.4121018888512731, 0.29283313687668966, -0.3446941769846983 |
| `zinb_theta_pkg`, `zinb_se_log_theta_pkg` | | 1.1216053960325127, 0.30655779164663227 |

Zero part on `x1` only (`zip_z1_*`): count 0.6072009458052031, 0.18265730366648034,
-0.30273324594357554; zero -0.3891251937775814, 0.43784369115847704; `zip_z1_se_zero`
0.12886042812630255, 0.13807038647154984; `zip_z1_loglik` -689.7078650175997. Missing
values (`y_na ~ x1 + x2_na | x1 + x2_na`, 495 rows): count 0.631486489779986,
0.19001742937939975, -0.2619236371428857; zero -0.3678354788343926, 0.43133493971040254,
0.19474884922867577; `zip_na_se_count` 0.054914842224500375, 0.05315774411900642,
0.060678332311512276; `zip_na_loglik` -684.0056763199633; 5 rows dropped (`y_na` 3,
`x2_na` 2). `hurdle_na_n_used` 495 (pscl::hurdle on the same data; [hurdle.md](hurdle.md)).
Zero part on its intercept alone (`zip_z0_*`, `y ~ x1 + x2 | 1`): count
0.60724894342877345, 0.11776749247011398, -0.31583962342782135; zero -0.35958412819172109;
`zip_z0_se_count` 0.055207787807567452, 0.050945419658198668, 0.054371358544598181;
`zip_z0_se_zero` 0.12295792255917738; `zip_z0_loglik` -695.3298205798277.

## Rust tests

R-pinned (`crates/p2a-core/src/econometrics/discrete/count_models.rs` unless noted):

| Test | Quantities |
|---|---|
| `econometrics::discrete::count_models::tests::test_validate_negbin_theta_disc_negbin` | `coef`, `se`, `z`, `p`, `theta`, `se_theta`, `loglik`, `aic`, `bic`, `y_mean`, `y_var`, `n`; cross-checks `coef_pkg`, `theta_pkg`, `se_theta_pkg`; converged, no warning |
| `econometrics::discrete::count_models::tests::test_validate_negbin_init_theta_is_a_starting_value` | `init_theta` 50, 0.2 and the range ends 1e-8 and 1e6 reach `theta` and `coef` (converged, no warning); 0, -1, NaN, inf, 1e-300, 1e-9, 2e6 and 1e300 are refused with a message naming `init_theta` and both bounds |
| `econometrics::discrete::count_models::tests::test_validate_negbin_missing_rows_dropped` | `y_na ~ x1_na + x2`: `na_*` estimates, 395 rows used, 5 dropped by column |
| `econometrics::discrete::count_models::tests::test_validate_negbin_trigamma_disc_negbin` | `trigamma` against R's at `trigamma_x` |
| `econometrics::discrete::count_models::tests::test_validate_negbin_intercept_only_disc_negbin` | `int_*` (`glm.nb(y ~ 1)`): coefficient, SE, theta, SE(theta), log-likelihood; converged, no warning |
| `p2a-mcp` `tier2::wp10_ordered_count::test_validate_mcp_negbin_disc_negbin` (`crates/p2a-mcp/tests/tier2/wp10_ordered_count.rs`) | the MCP `negbin` text: coefficients, SE, z, p, theta with SE, log-likelihood at print precision; `Converged: true` |
| `econometrics::discrete::count_models::tests::test_validate_zip_disc_zeroinfl` | every `zip_*` key above (count, zero, SEs, z, p, log-likelihood, AIC, BIC, n, zeros; `*_pkg` cross-checks); converged, no warning; no theta |
| `econometrics::discrete::count_models::tests::test_validate_zinb_disc_zeroinfl` | every `zinb_*` key, with theta, SE(log theta) and SE(theta) |
| `econometrics::discrete::count_models::tests::test_validate_zip_zero_part_regressors` | `zip_z1_*` (zero part on `x1`) |
| `econometrics::discrete::count_models::tests::test_validate_zip_missing_rows_dropped` | `zip_na_*`: estimates, 495 rows used, 5 dropped by column |
| `econometrics::discrete::count_models::tests::test_validate_zip_intercept_only_zero_part` | `zip_z0_*` (`y ~ x1 + x2 \| 1`): count and zero coefficients, SEs, log-likelihood |
| `econometrics::discrete::count_models::tests::test_validate_zeroinfl_regressor_scale_disc_zeroinfl` | ZIP and ZINB with x1 multiplied by 1e4, 1e6, 1e10 and 1e12: converged at the unscaled fit's iteration, no warning; x1 coefficients and SEs times the scale, all other coefficients, SEs, z, SE(log theta) and the log-likelihood equal the unscaled fit to 1e-7 relative, and pscl's `zip_*` / `zinb_*` values |
| `p2a-mcp` `tier2::wp10_ordered_count::test_validate_mcp_zeroinfl_disc_zeroinfl` | the MCP `zeroinfl` text for both distributions: both coefficient tables, log-likelihood, and the ZINB theta line, at print precision |

Not R-pinned:

- `econometrics::discrete::count_models::tests::test_negbin_without_overdispersion_is_flagged`:
  the underdispersed 12-row fixture gives theta above 1e6, `converged = false`,
  `SE(theta) = NaN` and the overdispersion warning.
- `econometrics::discrete::count_models::tests::test_negbin_all_zero_outcome_is_not_converged`:
  an all-zero outcome (no MLE), with 30 and with 5 rows, gives `converged = false` and the
  no-MLE warning, which counts every row.
- `econometrics::discrete::count_models::tests::test_negbin_separating_regressor_is_not_converged`:
  seeded NB counts with a dummy whose rows all have y = 0 (no MLE): `converged = false`,
  the dummy's coefficient below -10 and the no-MLE warning, which counts the dummy's 30
  rows.
- `econometrics::discrete::count_models::tests::test_negbin_separation_is_flagged_on_large_data`:
  seeded NB counts on 5000 rows with a dummy on every 50th row whose rows all have y = 0,
  where the IRLS stops with the separated means at 5e-14 to 1e-13, above 10 eps (a
  verdict that needed such means reported the fit as converged): `converged = false`, the
  dummy's coefficient below -10 and the no-MLE warning for its 100 rows.
- `econometrics::discrete::count_models::tests::test_negbin_high_leverage_zero_row_converges`:
  `disc_negbin`'s data plus the row `y = 0, x1 = -80, x2 = 0.1` (fitted mean far below
  10 eps): converged, a single warning, the note counting 1 row, and the coefficients and
  theta of the 400-row fit to 1e-6 relative (R's glm.nb: converged, theta 1.602585).
- `econometrics::discrete::count_models::tests::test_negbin_ill_conditioned_designs_converge`:
  seeded NB counts on two ill-conditioned designs, `y ~ x + x^2 + x^3` with
  `x ~ U(10, 20)` and `y ~ x1 + x2` with `x2 = x1 + 1e-4 noise`, converge without a
  warning, with finite SEs (glm.nb on the cubic sample agrees to 2e-10).
- `econometrics::discrete::count_models::tests::test_negbin_stops_when_cancelled`:
  a cancelled budget stops the fit.
- `p2a-mcp` `tier2::wp10_ordered_count::test_mcp_negbin_reports_dropped_rows_and_rejects_bad_init_theta`:
  the MCP text reports the dropped rows; a negative `init_theta` is a tool error.
- `econometrics::discrete::count_models::tests::test_zinb_without_overdispersion_is_flagged`:
  ZINB on underdispersed positive counts (the case's `y` with its positive values replaced
  by the rounded count mean): theta above 1e6, `converged = false`, NaN theta SEs, the
  warning naming pscl::zeroinfl's Poisson model; the count and zero parts, their SEs and
  the log-likelihood equal the ZIP fit on the same data to 1e-5 relative.
- `econometrics::discrete::count_models::tests::test_theta_newton_reports_a_theta_it_cannot_update`:
  at theta = 1e-300 (trigamma overflows) the theta step reports that it could not move.
- `econometrics::discrete::count_models::tests::test_zip_separated_zero_part_is_flagged`:
  the 12-row fixture with a separated zero part (pscl: zero coefficients 94.70464,
  -25.39549, logLik -8.370407) reaches at least pscl's log-likelihood and warns that
  fitted probabilities were numerically 0 or 1.
- `econometrics::discrete::count_models::tests::test_zinb_extreme_row_is_a_note`: ZINB on
  `disc_zeroinfl`'s `y_nb` plus the row `y_nb = 0, x1 = -90, x2 = 0`: converged, the one
  warning is glm.fit's diagnostic counting 1 row, and the estimates are those without the
  row to 1e-6 relative.
- `econometrics::discrete::count_models::tests::test_zeroinfl_analytic_derivatives_match_central_differences`:
  the analytic gradient and Hessian equal central differences of the log-likelihood and
  the gradient, both distributions, with the zero part on all and on fewer regressors.
- `econometrics::discrete::count_models::tests::test_zeroinfl_rejects_outcome_without_zeros`,
  `econometrics::discrete::count_models::tests::test_na_count_models_float_outcome_equals_int64`,
  `econometrics::discrete::count_models::tests::test_na_count_models_text_outcome_is_non_numeric`,
  `econometrics::discrete::count_models::tests::test_na_count_models_column_errors`,
  `econometrics::discrete::count_models::tests::test_zeroinfl_stops_when_cancelled`:
  the zero-count refusal, Int64 = Float64 outcomes, `NonNumericColumn` for a text outcome
  (NB, ZIP, hurdle), `ColumnNotFound` for an absent outcome, count regressor or zero-part
  regressor and `NonNumericColumn` for a text regressor in either part (NB, ZIP, ZINB,
  hurdle), cancellation.

## Tolerance

`Tol` (`crates/p2a-validation`), validation/README.md's iterative row: coefficients and
theta `Tol::iterative(1e-6)`, SEs, `SE(theta)` and z `Tol::iterative(1e-5)`, p
`Tol::iterative(2e-3)`, log-likelihood, AIC, BIC `Tol::stat(400)` (1e-6), mean and variance
of y `Tol::coef(400)` (1e-8), counts exact; glm.nb's `SE.theta` `Tol::iterative(1e-4)`;
trigamma `Tol::new(1e-12, 0)`; MCP text `Tol::printed(4)`. Zero-inflated: coefficients
and theta `Tol::iterative(1e-6)`, SEs (incl. SE(log theta), SE(theta)) and z
`Tol::iterative(1e-5)`, p `Tol::iterative(2e-3)` (z^2 is at most 130 here),
log-likelihood, AIC, BIC `Tol::stat(500)` (1e-6), counts exact, pscl's own optimum
`Tol::iterative(1e-4)`. Measured: the NB fit agrees with the polished values to about
1e-14 relative (coefficients, theta, SEs); ZIP to 1e-14 (coefficients) and 3e-11 (SEs);
ZINB to 2e-10 (coefficients) and 1.3e-9 (SEs; the R side takes them from
`numDeriv::hessian`, p2a from the analytic Hessian). The regressor-scale test holds the
scaled fits to the unscaled one at 1e-7 relative.

## Status

R-pinned, known deviations (ECON-64): registry
`validation/methods/discrete_ordered_count.json`, ids `discrete_negbin` (deviation
ECON-64), `discrete_zip` and `discrete_zinb`.

## Known Differences

- ECON-64 (Tier 4): `log_likelihood_null` omits the lnGamma terms and reuses the full
  model's theta, so it and `pseudo_r_squared` are not glm.nb's null model; not pinned.
- glm.nb's own `SE.theta` is evaluated one `theta.ml` step before its reported theta; it
  is checked at 1e-4.
- No overdispersion: theta stops at a value above 1e6 (bounded by 1e8) with
  `converged = false`, where glm.nb stops at the iteration limit.
- ZIP/ZINB: pscl's BFGS stops about 1e-7 relative from the MLE (`*_pkg`, checked at
  1e-4); p2a warns about fitted zero-inflation probabilities numerically 0 or 1, which
  pscl's EM suppresses, and refuses an outcome without zeros as pscl does.
- Open items: ECON-58 (Tier 4) keeps negative and non-integer counts and the `y < 0.5`
  zero rule; TOOL-30 (Tier 4) the MCP description of the `z` argument.
- `validation/expected/negbin_test.csv`, `zip_test.csv` and `zinb_test.csv` are legacy
  files that no test reads.

## References

- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.),
  section 7.4. Springer.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.).
  Cambridge University Press.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in
  manufacturing. *Technometrics*, 34(1), 1-14.
- Zeileis, A., Kleiber, C., & Jackman, S. (2008). Regression models for count data in R.
  *Journal of Statistical Software*, 27(8), 1-25.
