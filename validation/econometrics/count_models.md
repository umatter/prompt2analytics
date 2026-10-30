# Validation: Count Models (MASS::glm.nb, pscl::zeroinfl)

Registry: [`validation/methods/discrete_ordered_count.json`](../methods/discrete_ordered_count.json)
(method `discrete_negbin`). R case: `disc_negbin` in
[`validation/r/cases/discrete.R`](../r/cases/discrete.R). The hurdle models are in
[hurdle.md](hurdle.md).

## Method Overview

### Negative binomial

`run_negbin(dataset, y, x, init_theta)` fits the NB2 model of `MASS::glm.nb`:

    y ~ NB(mu, theta),   log(mu) = x'beta,   Var(y) = mu + mu^2 / theta

`init_theta` is the starting value of theta (glm.nb's `init.theta`); it must be positive
and finite (`InvalidSpecification` naming `init_theta` otherwise).

Algorithm (glm.nb's alternation):

- Start: the coefficients of a Poisson fit (glm's IRLS from `mu = y + 0.1`), or of the NB
  fit at `init_theta` when given; theta from `init_theta` or MASS's moment start
  `n / sum((y / mu - 1)^2)`.
- Coefficients: Fisher scoring (IRLS) at fixed theta, weights `W = mu theta / (theta + mu)`,
  working response `z = eta + (y - mu) / mu`, until `max |delta beta| < 1e-12 (1 + max
  |beta|)`. A design whose `X'WX` is not positive definite is `SingularMatrix`.
- theta: Newton steps on `ln theta` at fixed `mu` with MASS `theta.ml`'s score
  `s = sum(digamma(theta + y) - digamma(theta) + ln theta + 1 - ln(theta + mu) - (y + theta) / (mu + theta))`
  and information
  `I = sum(-trigamma(theta + y) + trigamma(theta) - 1/theta + 2/(mu + theta) - (y + theta)/(mu + theta)^2)`
  (MASS's signs), clamped to +-2 and halved while the profile log-likelihood falls.
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
| `trigamma` at `trigamma_x` = 0.1, 0.5, 1, 1.6025851429909947, 3.7, 6, 10, 25.5, 1000 | 101.43329915079275, 4.934802200544679, 1.6449340668482264, 0.8566167623849812, 0.31003785767003833, 0.18132295573711527, 0.10516633568168572, 0.03999466964956293, 0.0010005001666666335 |

## Rust tests

R-pinned (`crates/p2a-core/src/econometrics/discrete/count_models.rs` unless noted):

| Test | Quantities |
|---|---|
| `econometrics::discrete::count_models::tests::test_validate_negbin_theta_disc_negbin` | `coef`, `se`, `z`, `p`, `theta`, `se_theta`, `loglik`, `aic`, `bic`, `y_mean`, `y_var`, `n`; cross-checks `coef_pkg`, `theta_pkg`, `se_theta_pkg`; converged, no warning |
| `econometrics::discrete::count_models::tests::test_validate_negbin_init_theta_is_a_starting_value` | `init_theta` 50 and 0.2 reach `theta` and `coef`; 0, -1, NaN and inf are refused |
| `econometrics::discrete::count_models::tests::test_validate_negbin_missing_rows_dropped` | `y_na ~ x1_na + x2`: `na_*` estimates, 395 rows used, 5 dropped by column |
| `econometrics::discrete::count_models::tests::test_validate_negbin_trigamma_disc_negbin` | `trigamma` against R's at `trigamma_x` |
| `p2a-mcp` `tier2::wp10_ordered_count::test_validate_mcp_negbin_disc_negbin` (`crates/p2a-mcp/tests/tier2/wp10_ordered_count.rs`) | the MCP `negbin` text: coefficients, SE, z, p, theta with SE, log-likelihood at print precision; `Converged: true` |

Not R-pinned:

- `econometrics::discrete::count_models::tests::test_negbin_without_overdispersion_is_flagged`:
  the underdispersed 12-row fixture gives theta above 1e6, `converged = false`,
  `SE(theta) = NaN` and the overdispersion warning.
- `econometrics::discrete::count_models::tests::test_negbin_stops_when_cancelled`:
  a cancelled budget stops the fit.
- `p2a-mcp` `tier2::wp10_ordered_count::test_mcp_negbin_reports_dropped_rows_and_rejects_bad_init_theta`:
  the MCP text reports the dropped rows; a negative `init_theta` is a tool error.

## Tolerance

`Tol` (`crates/p2a-validation`), validation/README.md's iterative row: coefficients and
theta `Tol::iterative(1e-6)`, SEs, `SE(theta)` and z `Tol::iterative(1e-5)`, p
`Tol::iterative(2e-3)`, log-likelihood, AIC, BIC `Tol::stat(400)` (1e-6), mean and variance
of y `Tol::coef(400)` (1e-8), counts exact; glm.nb's `SE.theta` `Tol::iterative(1e-4)`;
trigamma `Tol::new(1e-12, 0)`; MCP text `Tol::printed(4)`. Measured: the Rust fit agrees
with the polished values to about 1e-14 relative (coefficients, theta, SEs).

## Status

R-pinned, known deviations (ECON-64): registry
`validation/methods/discrete_ordered_count.json`, id `discrete_negbin`.

## Known Differences

- ECON-64 (Tier 4): `log_likelihood_null` omits the lnGamma terms and reuses the full
  model's theta, so it and `pseudo_r_squared` are not glm.nb's null model; not pinned.
- glm.nb's own `SE.theta` is evaluated one `theta.ml` step before its reported theta; it
  is checked at 1e-4.
- No overdispersion: theta stops at a value above 1e6 (bounded by 1e8) with
  `converged = false`, where glm.nb stops at the iteration limit.
- `validation/expected/negbin_test.csv` is a legacy file that no test reads.

## References

- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.),
  section 7.4. Springer.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.).
  Cambridge University Press.
