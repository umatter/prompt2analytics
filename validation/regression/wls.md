# Validation: Weighted Least Squares

## Method Overview

`run_wls(dataset, y, x, weights, intercept, cov_type)` (MCP: `regression_ols` with `weights`)
fits `lm(y ~ x, weights = w)`: the coefficients minimise `sum(w_i (y_i - x_i'b)^2)`. It is OLS on
`sqrt(w) * y` and `sqrt(w) * X`, so the classical and HC0-HC3 covariances are those of
[ols_robust_se.md](ols_robust_se.md) on the transformed data, with the leverages of the weighted
fit. What changes from OLS:

- R² and F use the explained sum of squares `sum(w (f - m)^2)` about the weighted mean
  `m = sum(w f) / sum(w)` of the fitted values (`sum(w f^2)` without an intercept), as `summary.lm`.
- The log-likelihood adds `sum(log w) / 2` (`logLik.lm`); AIC and BIC follow.
- The residuals are `y - X b` on the original scale (`residuals.lm`).
- Weights must be finite and non-negative with at least one positive; a missing weight is refused.
  Rows with weight 0 are left out of the fit, with a warning: `lm` keeps them out of `nobs` and
  the residual degrees of freedom, and n, the HC1 factor and the leverages count only the rows with
  positive weight.

Matching weights (`propensity_matching`'s `weights` column) give MatchIt's recommended outcome model
on the matched sample: `lm(y ~ treat, data = match.data(m), weights = weights)`.

## Reference Implementation

`stats::lm(weights = )`, `summary.lm`, `logLik`, `AIC`, `BIC`, `sandwich::vcovHC` (HC0-HC3) and
`lmtest::waldtest(vcov = vcovHC(fit, "HC1"), test = "F")`, case `wls_sim` in
`validation/r/cases/regression.R` ([source](../r/cases/regression.R)): 300 simulated rows with
heteroskedastic noise and weights drawn from {0, 0.25, 1, 2.5, 4}, 51 of them 0.

With a weight of 0, `sandwich::vcovHC` mixes all n rows with `nobs`: HC0 and HC1 scale by the wrong
n, and HC2 and HC3 recycle the n residuals against the shorter vector of hat values (R warns
"longer object length is not a multiple of shorter object length"). The reference is therefore the
fit on the rows with positive weight; the case checks that it equals `lm`'s fit with the zero
weights in the coefficients, log-likelihood, R², `nobs` and residual degrees of freedom.

## Tests and tolerances

`test_validate_wls_sim` pins the coefficients, the classical and HC0-HC3 SEs, t, p and 95% CIs
(directly and through `LinearEstimator`), R², adjusted R², sigma, F with its df and p, the
log-likelihood, AIC, BIC, n, the residual df, the HC1 Wald F and its p, and the residuals of the
rows with positive weight; tolerances from `validation/README.md` for n = 249 (coefficients 1e-8,
SEs and statistics 1e-6, relative). `test_wls_checks_weights_and_unit_weights_are_ols` checks the
weight refusals and that unit weights give OLS. `test_validate_mcp_regression_ols_weights` checks
`regression_ols` with `weights` (printed coefficients and HC1 SEs, 4 decimals).
