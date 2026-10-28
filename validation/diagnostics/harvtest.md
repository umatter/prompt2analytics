# Validation: Harvey-Collier Test (harvtest)

## Method Overview

The Harvey-Collier test detects departures from linearity in regression models using recursive residuals. Under the null hypothesis of correct linear specification, the mean of recursive residuals should be zero.

**Key Features:**
- Uses one-step-ahead forecast errors (recursive residuals)
- Detects convex or concave functional misspecification
- Based on t-test of recursive residual mean

**Recursive Residuals:**
For observation t (t > k):
w_t = (y_t - x_t' β̂_{t-1}) / √(1 + x_t'(X_{t-1}'X_{t-1})⁻¹x_t)

where β̂_{t-1} is estimated using observations 1 to t-1.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| lmtest | R | `harvtest()` | 0.9.40 |
| strucchange | R | `recresid()` | 1.6.0 |

## R reference (pinned)

Two R reference cases pin the test: the concave DGP of the legacy script
`validation/scripts/validate_regression_diag.R` (Test 6), ported with the same seed and draws,
and the Grunfeld regression of the `p2a reg diagnostics` exhibit.

| Case | Data | R code |
|---|---|---|
| `diag_harvtest_nonlin` | `validation/datasets/generated/harvtest_nonlin.csv` (n = 100; `x = seq(1, 10, length.out = 100)`, `y = 5 + 2 x - 0.2 x^2 + N(0, 0.5^2)`, `set.seed(42)`), fitted linearly | `validation/r/cases/regression_legacy.R` |
| `ols_diagnostics_grunfeld` | `validation/datasets/grunfeld.csv` (n = 200; `inv ~ value + capital`) | `validation/r/cases/regression.R` |

R call: `harvtest(lm(y ~ x))` (lmtest 0.9.40); the signed mean and standard error of the
recursive residuals come from `strucchange::recresid` (1.6.0).

| Case | Quantity | R value |
|---|---|---|
| `diag_harvtest_nonlin` | \|t\| (df, p) | 8.454862195 (97, 2.871377594e-13) |
| | mean of the recursive residuals | -0.8521908541 |
| | standard error of that mean, `sd(w) / sqrt(98)` | 0.1007929916 |
| | number of recursive residuals | 98 |
| `ols_diagnostics_grunfeld` | \|t\| (p) | 1.469927271 (0.1431848398); df 196, 197 recursive residuals |

Tests: `test_validate_harvey_collier_nonlin` (every value above for `diag_harvtest_nonlin`, and
the sign of p2a's statistic against the sign of the mean) and
`test_validate_harvey_collier_grunfeld` (the Grunfeld row; df and the residual count are
asserted directly because the case does not store them).

## Property tests

- `test_harvey_collier_linear`
- `test_harvey_collier_quadratic`
- `test_harvey_collier_displays`
- `test_recursive_residuals_basic`

## Numerical Precision Summary

The tolerances of `validation/README.md` for n = 100 and 200:

- **t-statistic, mean and standard error of the recursive residuals**: 1e-6 relative
- **p-value**: 1e-6 relative or 1e-12 absolute
- **Degrees of freedom and residual count**: exact (df = n - k - 1)

The methods registry entry is `harvtest` in `validation/methods/regression_diagnostics.json`;
it records no deviation. These cases supersede the legacy
`validation/expected/harveycollier_test.csv`, whose values they reproduce to the 15 digits it
prints.

## Known Differences

1. **Observation ordering**: R allows `order.by` parameter for custom ordering; Rust uses data order
2. **Recursive computation**: Sequential matrix operations may accumulate small numerical differences
3. **Starting point**: Both start at observation k+1 (first observation with full degrees of freedom)
4. **Sign of the statistic**: p2a reports the signed t of the mean recursive residual;
   `harvtest` reports |t|. The p-value is the same two-sided value.
5. **Cost**: p2a refits the regression for every recursive residual, O(n^2 k^2) in total
   (REG-38, Tier 4).

## Implementation Notes

The Rust implementation:
1. Builds design matrix X with intercept
2. For each observation t > k:
   - Estimates OLS using observations 0 to t-1
   - Computes one-step-ahead forecast error
   - Standardizes by √(1 + x_t'(X'X)⁻¹x_t)
3. Performs t-test on mean of recursive residuals
4. Reports t-statistic, p-value, and interpretation

## References

- Harvey, A.C. & Collier, P. (1977). "Testing for Functional Misspecification
  in Regression Analysis." *Journal of Econometrics*, 6(1), 103-119.
- Brown, R.L., Durbin, J. & Evans, J.M. (1975). "Techniques for Testing the
  Constancy of Regression Relationships over Time." *Journal of the Royal
  Statistical Society: Series B*, 37(2), 149-192.
