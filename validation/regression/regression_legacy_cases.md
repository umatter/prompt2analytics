# R references for the legacy regression-diagnostics DGPs

The legacy script `validation/scripts/validate_regression_diag.R` simulates its data in R and
writes only the results (`validation/expected/*_test.csv`), so no Rust test could compute on the
same data. Its data-generating code (Tests 1, 2, 4-8 and 10) is ported verbatim, with the same
seeds and draws in the same order, into the R reference cases of
`validation/r/cases/regression_legacy.R`. Each case writes its data to
`validation/datasets/generated/<name>.csv`, reads it back, and computes its reference values on
the re-read file; `Rscript validation/r/run.R --check` recomputes every case and fails on any
difference. The case values equal the legacy CSVs to the 15 digits those print.

Each section below names the tests that pin a case and the quantities they leave out. The
quantities p2a does not reproduce are deviations, recorded with the R and the p2a value in
`validation/methods/regression_diagnostics.json`. R 4.3.3; the package versions are recorded in
each case file (`validation/expected/<case>.json`, key `packages`). The Breusch-Godfrey and
Harvey-Collier pins are documented in `validation/diagnostics/bgtest.md` and
`validation/diagnostics/harvtest.md`.

Tolerances are those of `validation/README.md`: for n = 100 to 1000, coefficients 1e-8, standard
errors and statistics 1e-6 relative; for n < 100, coefficients 1e-6 and standard errors 1e-5
relative; p-values at the standard-error tolerance or 1e-12 absolute; degrees of freedom and
counts exact.

## RESET (diag_reset_quad, ols_diagnostics_grunfeld)

DGP (`diag_reset_quad`, legacy Test 5): `set.seed(42)`, n = 100, `x ~ U(1, 10)`,
`y = 2 + 3 x + 0.5 x^2 + N(0, 2^2)`, fitted linearly (`y ~ x`); data
`validation/datasets/generated/reset_quad.csv`. The Grunfeld rows come from the
`ols_diagnostics_grunfeld` case of `validation/r/cases/regression.R` (`inv ~ value + capital`,
n = 200).

R call: `resettest(lm(y ~ x), power = 2:3, type = "fitted" | "regressor")` (lmtest); the
R-squared of the augmented models from `stats::lm`.

| Case | Quantity | R value |
|---|---|---|
| `diag_reset_quad` | fitted F (df1, df2, p) | 150.2461396 (2, 96, 2.714887975e-30) |
| | regressor F (df1, df2, p) | 150.2461396 (2, 96, 2.714887975e-30) |
| | R-squared: original, augmented (fitted), augmented (regressor) | 0.9752269565, 0.9940018701, 0.9940018701 |
| `ols_diagnostics_grunfeld` | fitted F (p) | 19.56011807 (1.811659783e-08); df (2, 195) |
| | regressor F (p) | 9.627880717 (4.044025645e-07); df (4, 193) |

With one regressor the fitted and regressor augmentations span the same columns, so the two
`diag_reset_quad` statistics agree to rounding.

Tests: `test_validate_reset_quad` (every `diag_reset_quad` value above, and n) and
`test_validate_reset_grunfeld` (the Grunfeld statistics and p-values; df asserted directly).
Tolerance: statistics and R-squared 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute,
df exact.

Deviation REG-35: `type = "princomp"` is not implemented. `ResetType::PrinComp` reuses the
fitted-value augmentation, so on Grunfeld p2a gives the fitted F, 19.56011807, where
`resettest(type = "princomp")` gives 6.108133142. No case exercises it.

## Wald test (diag_wald_sim, ols_diagnostics_grunfeld)

DGP (`diag_wald_sim`, legacy Test 7): `set.seed(42)`, n = 100, `x1, x2, x3 ~ N(0, 1)`,
`y = 2 + 3 x1 + 0.001 x2 + x3 + N(0, 0.5^2)`; data `validation/datasets/generated/wald_sim.csv`.
The unrestricted model is `y ~ x1 + x2 + x3`; the restrictions drop `x2`, and `x2` and `x3`.
Grunfeld: `inv ~ value + capital` against `inv ~ value`.

R call: `waldtest(m_unrestricted, m_restricted, test = "F" | "Chisq")` (lmtest, classical
covariance). The legacy script used `aod::wald.test(b = coef(m), Sigma = vcov(m), Terms = ...)`,
which equals the Chisq form; the case records both and checks that they agree.

| Case | Quantity | R value |
|---|---|---|
| `diag_wald_sim` | drop x2: F (df1, df2, p) | 0.01198882273 (1, 96, 0.9130396463) |
| | drop x2 and x3: F (df1, df2, p) | 247.1956558 (2, 96, 1.362389280e-38) |
| | RSS: unrestricted, drop x2, drop x2 and x3 | 18.87075804, 18.87311469, 116.0534541 |
| | parameters: unrestricted, drop x2, drop x2 and x3 | 4, 3, 2 |
| `ols_diagnostics_grunfeld` | drop capital: F (p) | 81.98954629 (1.347370105e-16); df (1, 197) |

Tests: `test_validate_wald_sim` (every `diag_wald_sim` value above, and n) and
`test_validate_wald_grunfeld` (the Grunfeld F and p; df asserted directly). Tolerance:
statistics and RSS 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute, counts exact.

Deviation REG-37: p2a's Chisq form is n (RSS_r - RSS_u) / RSS_u; lmtest's and aod's is
q F = (n - k) (RSS_r - RSS_u) / RSS_u. It is left out of both tests.

| Restriction | R Chisq (`waldtest`, `aod::wald.test`) | p2a Chisq |
|---|---|---|
| `diag_wald_sim`, drop x2 | 0.01198882273 | 0.01248835701 |
| `diag_wald_sim`, drop x2 and x3 | 494.3913116 | 514.9909496 |
| Grunfeld, drop capital | 81.98954629 | 83.23811806 |

## Legacy files superseded

| Legacy file (`validation/expected/`) | Superseded by | Note |
|---|---|---|
| `bgtest_test.csv` | `diag_bgtest_ar1` | values equal to 15 digits |
| `reset_test.csv` | `diag_reset_quad` | values equal to 15 digits |
| `harveycollier_test.csv` | `diag_harvtest_nonlin` | values equal to 15 digits |
| `wald_test.csv` | `diag_wald_sim` | `p_value` is `NA`: `aod::wald.test` names its p-value `P`, not `P(> Chi2)`; the case records it |
