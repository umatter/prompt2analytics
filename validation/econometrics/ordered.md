# Validation: Ordered Logit and Probit (MASS::polr)

Registry: [`validation/methods/discrete_ordered_count.json`](../methods/discrete_ordered_count.json)
(methods `discrete_ordered_logit`, `discrete_ordered_probit`). R case: `disc_ordered` in
[`validation/r/cases/discrete.R`](../r/cases/discrete.R). The 5-, 16- and 30-category
fits of `disc_polr_j*` are in [discrete_newton.md](discrete_newton.md).

## Method Overview

`run_ordered_model_with(dataset, y, x, &OrderedConfig { link, levels, max_iter, tol })`
fits the proportional odds model of `MASS::polr`. `run_ordered_logit` and
`run_ordered_probit` call it with the default options (`max_iter` 100, `tol` 1e-10) and
the logit or probit link. For an outcome with categories `1 < 2 < ... < J`:

    P(y <= j | x) = F(zeta_j - x'beta),   j = 1, ..., J - 1

with `F` the logistic (logit) or standard normal (probit) CDF. There is no intercept:
it is absorbed into the cut points `zeta_1 < ... < zeta_{J-1}`, which are reported as
`thresholds` and labelled `categories[i]|categories[i+1]`, polr's names.

### Estimation

Newton-Raphson on `theta = (beta, zeta)` with the analytic score and observed Hessian.
Per observation in category `c` with `u = zeta_c - x'beta` and `l = zeta_{c-1} - x'beta`
(infinite for the first and last category), `p = F(u) - F(l)` (computed from the upper
tails when `l > 0`):

    dp/dbeta = x (f(l) - f(u)),  dp/dzeta_c = f(u),  dp/dzeta_{c-1} = -f(l)
    d2p/dbeta dbeta' = x x' (f'(u) - f'(l)),  d2p/dbeta dzeta_c = -x f'(u),
    d2p/dbeta dzeta_{c-1} = x f'(l),  d2p/dzeta_c^2 = f'(u),  d2p/dzeta_{c-1}^2 = -f'(l)
    score += dp / p,   Hessian += d2p / p - dp dp' / p^2

accumulated over the slopes and the (at most two) cut points of each observation. The R
case asserts that this analytic Hessian equals `numDeriv::hessian` of the log-likelihood.

- Start: the null-model MLE, `beta = 0`, `zeta_j = F^-1(P_j)` with `P_j` the cumulative
  category shares, so the first log-likelihood is the null one.
- Step: `(-H + lambda I)^-1 g` by Cholesky, `lambda = 0` unless `-H` is not positive
  definite (then Levenberg damping `1e-8 max(1, max diag(-H))`, ten times larger per
  attempt, at most 20).
- Line search: the step is halved (at most 30 times) until the cut points are strictly
  increasing and the log-likelihood does not fall.
- Converged: the Newton decrement `g'(-H)^-1 g` (from the step just computed) at most
  `1e-12 (1 + |ll|)`, relative change of the log-likelihood below `tol`, and every
  Newton-step component at most `1e-10 (1 + |theta_k|)`; when the step had to be damped,
  also every score component below `1e-8 (1 + |ll|)`. The decrement and the
  log-likelihood change do not depend on the units of a regressor: with x multiplied by
  `s` the score component of its slope is multiplied by `s`, and a test on `max |score|`
  alone (the earlier `max |score| < 1e-8 (1 + |ll|)`) failed at `s = 1e12` on a fit whose
  iterates were the unscaled ones. The step test is relative only for parameters above 1
  in absolute value (absolute below), so a change of units that makes a coefficient large
  tightens it and can cost an iteration (x divided by 1e6: converged, possibly an
  iteration later). A damped step `(-H + lambda I)^-1 g` shrinks with `lambda`, and so do
  its decrement and its components, so away from the optimum tiny damped steps could pass
  the three tests; the score test, which does depend on units, guards that case. At a
  regular optimum `-H` is positive definite and the step is not damped. A non-finite step
  or score, a failed line search or the iteration limit end the fit with
  `converged = false` and a message in `warnings` (never an error); that message still
  reports `max |score|`.
- Standard errors: square roots of the diagonal of the inverse observed information at the
  estimate (NaN with a warning if it is not positive definite); `z = b / SE`,
  `p = 2 P(Z > |z|)` from the upper tail.
- `log_likelihood_null = sum_j n_j ln(n_j / n)` (exact), McFadden pseudo-R2
  `1 - ll / ll_null`, AIC and BIC with `k + J - 1` parameters.

## Category order and `levels`

The categories follow the outcome's data type, as R's `factor()` orders them:

- integer and float columns by value (`-2 < -1 < 0 < ... < 10`, not the string order
  `"-1" < "-2"` or `"10" < "2"`); labels are the decimal labels (`"1"`, `"-2"`, `"1.5"`);
- an Enum column in its declared order;
- booleans `FALSE < TRUE`, dates by time.

A text (`String`) or `Categorical` outcome has no order the program could know. Without
`OrderedConfig::levels` the call fails with `InvalidSpecification`, naming the column and
listing the observed labels (at most 20). With `levels`, lowest category first, the
categories are exactly `levels`. `levels` must name every observed label exactly once; a
label without observations is refused, because polr cannot estimate the cut points next to
an empty category. R sorts text labels alphabetically (`factor()`), which for
`low/mid/high` is the order `high < low < mid`, a different model; p2a asks instead of
guessing (ECON-19). The MCP `ordered_model` tool and the CLI do not expose `levels` yet
(T2-WP18): a text outcome there needs a numeric or Enum recode.

At most 100 categories are accepted (a method-level cap on the dense Hessian).

## Missing values

The outcome and the regressors are read in one request with joint listwise deletion, as
polr's default `na.action = na.omit`. The dropped rows are reported in
`OrderedResult::missing` (JSON key `missing`, omitted when nothing was dropped) and in
the last line of the text output, for example
`Missing values: 7 of 400 rows dropped because of missing values (y_na: 4, x1_na: 3)`.
A missing outcome is never read as 0 or as an empty category. A text regressor is
`NonNumericColumn`, an unknown column `ColumnNotFound`.

## Reference Implementations

| Package | Version | Function |
|---|---|---|
| MASS | 7.3.60.0.1 (R 4.3.3) | `polr(method = "logistic" \| "probit", Hess = TRUE)` |
| numDeriv | 2016.8.1.1 | `hessian`, `jacobian` (polishing and the Hessian check) |

## R reference

Case `disc_ordered` (`validation/r/cases/discrete.R`; dataset
`validation/datasets/generated/ordered.csv`, 400 rows, seed 2026): `y` coded 1/2/3
(Int64), `y_label` the same categories as text `low/mid/high`, regressors `x1`, `x2`,
`y_wide` with 13 integer levels -2..10, and `y_na`, `x1_na` with 4 and 3 missing cells
(7 incomplete rows). `Rscript validation/r/run.R --write disc_ordered` writes it.

polr's optimiser (BFGS) stops with a gradient of about 2e-6 even at `reltol = 1e-15`, so
the case polishes polr's fit by Newton steps on the analytic score until
`max |grad| < 1e-9` (reached: logit 2.4e-14, probit 7.8e-15) and takes the standard
errors from the analytic observed information at that optimum. polr's own values are kept
as `*_pkg`.

```r
f <- polr(frm, data = d, method = link, Hess = TRUE, control = list(reltol = 1e-15, maxit = 10000))
par <- .polish(c(coef(f), f$zeta), function(p) .polr_grad(p, X, y, J, link))
se <- sqrt(diag(solve(-.polr_hess(par, X, y, J, link))))
ll0 <- sum(tab * log(tab / n))   # exact null log-likelihood
```

| key | logit | probit |
|---|---|---|
| `*_coef` x1, x2 | 0.7176113258294506, -0.5364924530218471 | 0.4318892927168739, -0.3261172143214279 |
| `*_zeta` 1\|2, 2\|3 | -0.3104999989491347, 0.7751482406047268 | -0.1885541185105889, 0.46930809751006763 |
| `*_se_coef` | 0.10909697897328377, 0.10249374102531485 | 0.06382964499296677, 0.06048652815672785 |
| `*_se_zeta` | 0.10865205155127043, 0.11398888293711927 | 0.0662150461417083, 0.06816815611356541 |
| `*_z` | 6.577737830899818, -5.23439234098538 | 6.766280664171996, -5.391567746728975 |
| `*_p` | 4.776594634602565e-11, 1.6552851689641808e-07 | 1.3213510493701722e-11, 6.984561696227597e-08 |
| `*_loglik` | -387.8375169818051 | -387.67661567185417 |
| `*_loglik_null` | -425.8062619900378 | -425.8062619900378 |
| `*_pseudo_r2` | 0.08916906207715902 | 0.08954693653395762 |
| `*_aic`, `*_bic` | 783.6750339636102, 799.6408921520422 | 783.3532313437083, 799.3190895321403 |
| `*_coef_pkg` (polr's BFGS) | 0.7176113394089276, -0.5364924622789209 | 0.4318892933801122, -0.32611721333299093 |
| `*_se_coef_pkg` | 0.10909698234585462, 0.10249374445634789 | 0.06382964603769874, 0.0604865289669144 |

Other fits of the case (logit):

| key | R value |
|---|---|
| `logit_counts` (y 1/2/3) | [174, 90, 136] |
| `alpha_labels_coef` (`levels = [high, low, mid]`) | -0.3008023752009304, 0.19936916875019578 |
| `alpha_labels_zeta` high\|low, low\|mid | -0.6325727186787082, 1.3220490783196688 |
| `alpha_labels_loglik` | -418.6363593973833 |
| `wide_levels` | [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] |
| `wide_counts` | [31, 31, 31, 30, 31, 31, 30, 31, 31, 30, 31, 31, 31] |
| `wide_coef` | 0.7040629941207036, -0.5343755058354243 |
| `wide_se_coef` | 0.09438675996201279, 0.09125522236972927 |
| `wide_loglik` | -981.2842944831065 |
| `na_coef` (x1_na, x2; 393 rows) | 0.7065573458170662, -0.5248096959746141 |
| `na_zeta` | -0.2996099050295564, 0.7912171394198321 |
| `na_se_coef` | 0.10890568760731159, 0.10305826886311184 |
| `na_loglik` | -381.72558129269845 |
| `na_n_used`, `na_n_dropped` | 393, 7 |
| `na_missing_y`, `na_missing_x1` | 4, 3 |

The case also holds the 12 cut points of `y_wide` (`wide_zeta`, named `-2|-1` ... `9|10`).

## Rust tests

R-pinned (`crates/p2a-core/src/econometrics/discrete/ordered_models.rs` unless noted):

| Test | Quantities |
|---|---|
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_disc_ordered` | every `logit_*` key above, `converged`, no warning |
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_probit_disc_ordered` | every `probit_*` key above |
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_text_outcome_with_levels` | `y_label` with `levels = [low, mid, high]` gives the integer fit (`logit_coef`, `logit_zeta`, `logit_se_coef`, `logit_loglik`); `[high, low, mid]` gives `alpha_labels_*` |
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_integer_levels_in_numeric_order` | `y_wide`: categories `-2, ..., 10` in numeric order, `wide_counts`, `wide_coef`, `wide_zeta`, `wide_se_coef`, `wide_loglik` |
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_enum_outcome_uses_declared_order` | `y_label` as an Enum `low < mid < high`: `logit_counts`, `logit_coef`, `logit_zeta`, `logit_loglik` |
| `econometrics::discrete::ordered_models::tests::test_validate_ordered_logit_missing_rows_dropped` | `y_na ~ x1_na + x2`: 393 rows used, 7 dropped (`y_na` 4, `x1_na` 3), the note; `na_coef`, `na_zeta`, `na_se_coef`, `na_loglik` |
| `p2a-mcp` `tier2::wp10_ordered_count::test_validate_mcp_ordered_model_disc_ordered` (`crates/p2a-mcp/tests/tier2/wp10_ordered_count.rs`) | the MCP `ordered_model` text for both links: coefficients, SE, z, p, cut points with SE, log-likelihood at print precision; `Converged: true` |

Not R-pinned (error paths, derivatives and fixtures):

- `econometrics::discrete::ordered_models::tests::test_ordered_analytic_derivatives_match_central_differences`:
  the private score and Hessian equal central differences of the log-likelihood and the
  score, both links.
- `econometrics::discrete::ordered_models::tests::test_ordered_text_outcome_requires_levels`:
  a text outcome without `levels` is an error that lists `high, low, mid` and asks for
  `levels`.
- `econometrics::discrete::ordered_models::tests::test_ordered_levels_are_validated`:
  a missing, unobserved, repeated or empty `levels` is refused; `[low, mid, high]` gives
  the counts of the integer fit.
- `econometrics::discrete::ordered_models::tests::test_na_ordered_float_outcome_equals_int64`:
  the same codes as Int64 and Float64 give the same fit.
- `econometrics::discrete::ordered_models::tests::test_na_ordered_column_errors`:
  `NonNumericColumn`, `ColumnNotFound` and no regressor.
- `econometrics::discrete::ordered_models::tests::test_ordered_accepts_100_categories`:
  100 categories of one row each converge to polr's MLE (inline R values).
- `econometrics::discrete::ordered_models::tests::test_ordered_config_is_validated`:
  `max_iter = 0` and a `tol` that is NaN, 0 or negative are `InvalidSpecification`
  naming the field.
- `econometrics::discrete::ordered_models::tests::test_ordered_iteration_limit_is_reported`:
  `max_iter = 2` ends the fit with `converged = false` and a "did not converge" warning.
- `p2a-mcp` `tier2::wp10_ordered_count::test_mcp_ordered_model_text_outcome_asks_for_levels`:
  the MCP tool error for a text outcome mentions `levels`.

## Tolerance

`Tol` (`crates/p2a-validation`), validation/README.md's iterative row: coefficients and
cut points `Tol::iterative(1e-6)`; SEs and z `Tol::iterative(1e-5)`; p-values
`Tol::iterative(2e-3)` (d ln p / d ln z is about z^2, at most 46 here, times the z
tolerance); log-likelihood, null log-likelihood, pseudo-R2, AIC, BIC `Tol::stat(400)`
(1e-6); counts exact; polr's own optimum (`*_pkg`) `Tol::iterative(1e-4)`; MCP text
`Tol::printed(4)`. Measured: the Rust fit agrees with the polished values to about 1e-11
relative (coefficients, cut points) and 1e-12 (SEs).

## Status

R-pinned (registry `validation/methods/discrete_ordered_count.json`, ids
`discrete_ordered_logit` and `discrete_ordered_probit`).

## Known Differences

- polr's BFGS stops about 1e-7 relative from the MLE; its values are recorded as `*_pkg`
  and checked at 1e-4.
- Text outcomes need `levels`, where R sorts the labels alphabetically (ECON-19).
- `validation/expected/ordered_logit_test.csv`, `ordered_logit_multi_test.csv` and
  `ordered_probit_test.csv` are legacy files that no test reads.

## References

- McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal
  Statistical Society B*, 42(2), 109-142.
- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.),
  section 7.3. Springer.
