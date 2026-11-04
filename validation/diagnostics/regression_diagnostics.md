# Validation: Regression Diagnostics

## Method Overview

`run_diagnostics(dataset, y, x)` fits OLS with an intercept and reports what
`p2a reg diagnostics` prints:

- **Jarque-Bera**: normality of the residuals, JB = (n/6)(S² + (K − 3)²/4) on χ²(2)
- **Breusch-Pagan**: heteroskedasticity, studentized (Koenker) form n·R² of the
  regression of û² on X, on χ²(r − 1), r the rank of that regression (k on a full-rank
  design)
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

## RESET type = princomp (T3-WP09)

`ResetType::PrinComp` follows `lmtest::resettest(type = "princomp")`: Z = the non-constant
design columns, v₁ the eigenvector of cov(Z) (n − 1 denominator) with the largest
eigenvalue, and the augmentation is powers of pc₁ = Z v₁, **not centred**. The sign of v₁
does not change the statistic. Until T3-WP09 the type reused the fitted-value
augmentation (REG-35). The text output names the augmentation (`Augmentation: powers of
the first principal component of the regressors (type = "princomp")`).

Case `reset_princomp` (`validation/r/cases/reg_rank_qr.R`, n = 150, `lm(y ~ x1 + x2)` with
correlated regressors and a quadratic term):

| Quantity | R |
|---|---|
| princomp, power 2:3 | F 106.8301274675919, df (2, 145), p 3.0524202840387294e-29 |
| fitted, power 2:3 | F 91.53815584456794, p 1.954347586092798e-26 |
| regressor, power 2:3 | F 41.51476192215581, df (4, 143) |
| princomp, power 3 | F 22.918632127631003, df (1, 146), p 4.102305695990578e-06 |
| power 3 with centred scores (recorded only) | 6.540979982413889 |

The power-3 pair shows the case distinguishes resettest's uncentred pc₁ from centred
scores. `test_validate_reset_princomp` pins every row (statistics 1e-6 relative, p-values
also 1e-12 absolute, df exact), checks `reset_test_from_ols` against `reset_test`, and
checks the component by its properties (unit norm, cov(Z) v = λ_max v within 1e-10,
scores = Z v, the statistic with pc₁ negated equal within 1e-10). Measured: statistics
within 4.0e-15 relative, p-values within 6.9e-14 relative.

## Wald test of nested models (T3-WP09)

`wald_test` ignores repeated names (the first is kept, with a warning "'<name>' is listed
more than once in x_unrestricted; duplicates are ignored"), as R's formula interface
does, and requires the restricted variables to be a strict subset of the unrestricted
ones. `wald_test_from_ols` counts the restrictions by rank, q = rank_u − rank_r, as
`anova.lm` does, so an aliased or collinear extra regressor adds no restriction; q = 0 is
`InvalidSpecification`. df2 = n − rank_u. Before (REG-36), q was the difference in name
counts, a repeated name gave F = 0, and swapped or longer restricted lists underflowed.

| Test | Case | R |
|---|---|---|
| `test_validate_wald_rejects_duplicates` | `ols_diagnostics_grunfeld` | `waldtest(m, m_value)` with capital listed twice: F 81.98954629331581, df (1, 197) |
| `test_validate_wald_aliased_rank_q` | `ols_aliased` | `anova(lm(y ~ x1), lm(y ~ x1 + x2 + x3))`, x2 = 2 x1: F 11.38526696929829, df (1, 117), p 0.001004720010233427 |

Registry entry `waldtest_nested_rank` (`validation/methods/reg_rank_qr.json`). Measured:
F within 3.9e-15 relative, p within 9.6e-14. The χ² form remains n (RSS_r − RSS_u)/RSS_u
(REG-37, Tier 4), not lmtest's.

## Diagnostics on aliased models (T3-WP09)

The auxiliary regressions of Breusch-Pagan, VIF, Breusch-Godfrey and RESET use
`lm.fit`'s pivoted QR, and the fitted values are y − residuals of the model's own fit.
Degrees of freedom count ranks:

- Breusch-Pagan: r − 1, r the rank of the regression of û² on X (as `bptest`);
- Breusch-Godfrey F: df2 = n − rank([X, lags]) (n − k − p on a full-rank design); the χ²
  form is unchanged;
- RESET: df1 = rank([X, Z]) − rank(X), df2 = n − rank([X, Z]), as `anova.lm` of the
  augmented fit; terms that alias are dropped with a warning ("RESET: a of q augmentation
  terms are collinear with the model and were dropped"), and an augmentation that adds
  nothing is `InvalidSpecification`. The sample-size check made before the augmented
  design is built counts ranks too (n > rank(X) + q), so an aliased column costs no
  observation; df2 > 0 is the binding check.

`lmtest::resettest` counts ncol(X) and every augmentation column instead (df2 = 113 on
the dummy-trap model, where anova gives 114). p2a keeps the rank count; the registry
records it as a deviation (`resettest_princomp`, REG-4). On full-rank designs the two
agree.

Pinned to `ols_aliased`, `y ~ x1 + da + db + dc` (dc aliased):

| Test | Quantity | R |
|---|---|---|
| `test_validate_diagnostics_aliased_model` | `bgtest(order = 1)` | LM 0.40333656734767115, p 0.5253711258846874 |
| | `bptest` (df 3) | 1.3783997498637213, p 0.7106056688722927 |
| `test_validate_reset_aliased_model` | anova of the RESET augmentation (power 2:3) | F 0.4446881921064504, df (2, 114), p 0.64213126502665 |

Measured: statistics within 1.6e-13 relative, p-values within 6e-14.
`test_reset_collinear_augmentation_df_by_rank` (a 0/1 regressor, type = regressor:
d² = d³ = d) checks df1 = 2 of 4 terms. `recursive_residuals` (Harvey-Collier) starts at
the first prefix of full rank and updates a QR factor of the prefix by Givens rotations,
O(k²) per observation; `test_recursive_residuals_match_prefix_refits` checks every residual
against a per-prefix least-squares refit within 1e-10.

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
