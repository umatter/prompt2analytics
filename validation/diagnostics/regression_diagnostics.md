# Validation: Regression Diagnostics

## Method Overview

`run_diagnostics(dataset, y, x)` fits OLS with an intercept and reports what
`p2a reg diagnostics` prints:

- **Jarque-Bera**: normality of the residuals, JB = (n/6)(S² + (K − 3)²/4) on χ²(2)
- **Breusch-Pagan**: heteroskedasticity, studentized (Koenker) form n·R² of the
  regression of û² on X, on χ²(k − 1)
- **Durbin-Watson**: first-order autocorrelation, Σ(ûₜ − ûₜ₋₁)²/Σûₜ²
- **VIF**: variance inflation factor of each regressor
- **Condition number**: of the unscaled design matrix X (intercept column included)

The Breusch-Godfrey, RESET, Harvey-Collier and Wald tests have their own pages
([bgtest.md](bgtest.md), [harvtest.md](harvtest.md)).

## Reference Implementation

| Quantity | R |
|---|---|
| Jarque-Bera | `tseries::jarque.bera.test(residuals(m))` (tseries 0.10.63) |
| Breusch-Pagan | `lmtest::bptest(m)` (studentized, the default; lmtest 0.9.40) |
| Durbin-Watson | `lmtest::dwtest(m)` |
| VIF | `car::vif(m)` (car 3.1.5) |
| Condition number | `kappa(model.matrix(m), exact = TRUE)` |

The case `ols_diagnostics_grunfeld` in `validation/r/cases/regression.R`
([source](../r/cases/regression.R)) computes them for `lm(inv ~ value + capital)` on
`validation/datasets/grunfeld.csv` (n = 200). It also holds the `lmtest` references for
`bgtest` (orders 1 and 4, χ² and F), `resettest` (fitted and regressor, powers 2:3),
`harvtest` and `waldtest` (dropping capital, F and χ²), which T1-WP04 pins, and the
non-studentized Breusch-Pagan statistic and the Durbin-Watson p-value, which p2a does
not compute. `Rscript validation/r/run.R --check` recomputes the case.

| Quantity | R | p2a |
|---|---|---|
| Jarque-Bera | 89.57176103371853 | 89.57176103371741 |
| Jarque-Bera p | 0 | 3.546e-20 |
| Breusch-Pagan (studentized, df 2) | 55.24708421216683 | 55.247084212166776 |
| Breusch-Pagan p | 1.0075070474936277e-12 | 1.0075273948473296e-12 |
| Durbin-Watson | 0.35818526971627723 | 0.3581852697162772 |
| VIF value, capital | 1.313772795649384, 1.313772795649384 | 1.3137727956493839, 1.3137727956493854 |
| Condition number | 2457.5938073301068 | 2457.5938073301154 |

## Tests

`test_validate_diagnostics_grunfeld` (`crates/p2a-core/src/regression/diagnostics.rs`,
registry entry `regression_diagnostics_exhibit` in `validation/methods/regression.json`)
pins JB and its p-value, BP and its p-value, DW, the two VIFs (by name) and the condition
number, and asserts n = 200 and three parameters. Tolerance (`validation/README.md`,
n = 200): statistics 1e-6 relative; p-values 1e-6 relative or 1e-12 absolute.

Measured agreement: every statistic within 1.3e-14 relative (JB), the p-values within
3.6e-20 absolute. Scaling JB by 1.00001, or returning the non-studentized BP statistic
(142.0221637191518, which equals R's `bptest(m, studentize = FALSE)`), makes the test fail.

Not pinned to R (property tests): `test_diagnostics_basic`, `test_durbin_watson`,
`test_jarque_bera`.

## Known Differences

1. **Condition number**: p2a reports the condition number of the unscaled X, which equals
   `kappa(model.matrix(m), exact = TRUE)`; the Belsley interpretation labels attached to it
   are REG-34 (Tier 4).
2. **VIF of the intercept**: p2a lists an `(Intercept)` row with VIF NaN; `car::vif` has
   no such row, and the test compares the regressors only (T1-WP05 drops the row from the
   text output).
3. **Durbin-Watson**: no p-value is computed (R's `dwtest` gives 2.1e-32 here).
4. **Breusch-Pagan**: only the studentized version is offered.
5. **Jarque-Bera p-value**: p2a uses the closed form exp(−JB/2) of the χ²(2) survival
   function, `tseries` uses 1 − pchisq, which is 0 here; they agree within 1e-12 absolute.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- regression::diagnostics::tests::test_validate_diagnostics_grunfeld
```

## References

- Jarque, C.M. & Bera, A.K. (1980). "Efficient Tests for Normality, Homoscedasticity and Serial Independence of Regression Residuals". *Economics Letters*, 6(3), 255-259.
- Breusch, T.S. & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity and Random Coefficient Variation". *Econometrica*, 47(5), 1287-1294.
- Koenker, R. (1981). "A Note on Studentizing a Test for Heteroscedasticity". *Journal of Econometrics*, 17(1), 107-112.
- Durbin, J. & Watson, G.S. (1950). "Testing for Serial Correlation in Least Squares Regression: I". *Biometrika*, 37(3/4), 409-428.
- Belsley, D.A., Kuh, E. & Welsch, R.E. (1980). *Regression Diagnostics*. Wiley.
