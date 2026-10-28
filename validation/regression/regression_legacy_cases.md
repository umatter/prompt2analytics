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

## HAC (hac_ar1)

DGP (legacy Test 8): `set.seed(42)`, n = 100, AR(1) errors `e[i] = 0.6 e[i - 1] + N(0, 1)`,
`x ~ N(0, 1)`, `y = 2 + 3 x + e`; data `validation/datasets/generated/hac_ar1.csv`. Model
`lm(y ~ x)`. p2a's default bandwidth for n = 100 is `floor(4 (n/100)^(2/9))` = 4 (`bw_rule`).

R calls (sandwich 3.1.3): `kernHAC(m, kernel = <k>, bw = 4, prewhite = FALSE, adjust = FALSE)`
for every kernel, also with `adjust = TRUE`; `kernHAC(m, kernel = "Bartlett", bw = 5, ...)`;
`NeweyWest(m, lag = 4, prewhite = FALSE | TRUE)`; and, for the legacy rows, `NeweyWest(m)` and
`vcovHAC(m)`. kernHAC weights lag j by k(j / bw).

Which R call each p2a kernel equals, at p2a bandwidth m:

| p2a kernel (m = 4) | p2a weight of lag j | Equals | Pinned |
|---|---|---|---|
| `bartlett` | 1 - j/(m + 1) | `NeweyWest(lag = m, prewhite = FALSE)` = `kernHAC(kernel = "Bartlett", bw = m + 1)` | yes |
| `parzen` | Parzen(j/m) | `kernHAC(kernel = "Parzen", bw = m)` | yes |
| `truncated` | 1 for j <= m | `kernHAC(kernel = "Truncated", bw = m)` | yes |
| `tukey-hanning` | Tukey-Hanning(j/m) | `kernHAC(kernel = "Tukey-Hanning", bw = m)` | yes |
| `qs` | QS(j/m), truncated at j = m | none: kernHAC's QS uses every lag | no (REG-50) |

R values (standard errors; covariance matrices row-major):

| Key | (Intercept) | x |
|---|---|---|
| `coef` | 2.084173867 | 3.003896817 |
| `neweywest_lag4` = `kernhac_bartlett_bw5` | 0.2204664393 | 0.1429954779 |
| `neweywest_lag4_vcov` | 0.04860545086, 0.005265828526 | 0.005265828526, 0.02044770671 |
| `kernhac_parzen_bw4` | 0.1957529711 | 0.1418580397 |
| `kernhac_parzen_bw4_vcov` | 0.03831922570, 0.004648265592 | 0.004648265592, 0.02012370342 |
| `kernhac_truncated_bw4` | 0.2574611959 | 0.1282680350 |
| `kernhac_truncated_bw4_vcov` | 0.06628626740, 0.006057068549 | 0.006057068549, 0.01645268880 |
| `kernhac_tukey_hanning_bw4` | 0.2139224548 | 0.1417968863 |
| `kernhac_tukey_hanning_bw4_vcov` | 0.04576281667, 0.005394890914 | 0.005394890914, 0.02010635695 |
| `neweywest_default` (legacy row "NeweyWest") | 0.2877355266 | 0.1353875952 |
| `vcovhac_default` (what the legacy rows "Bartlett" and "Parzen" held before INF-16 was fixed) | 0.1903233734 | 0.1422489769 |

Test: `test_validate_hac_ar1` pins the coefficients, the default bandwidth, and the standard
errors and covariance matrices of the four pinned kernels at m = 4; it also checks that the
default bandwidth gives the m = 4 Bartlett result and that no bandwidth warning fires.
Tolerance: coefficients 1e-8, standard errors and covariance entries 1e-6 relative, bandwidth
exact.

Deviations (left out of the test, values in the registry):

| Finding | Quantity | R | p2a |
|---|---|---|---|
| REG-50 | QS kernel, m = 4 (`kernhac_qs_bw4`) | 0.2300539548, 0.1392422079 | 0.2284575999, 0.1414513262 |
| REG-50 | Bartlett at `kernHAC(bw = 4)` (`kernhac_bartlett_bw4`): p2a's m means bw = m + 1 | 0.2102028704, 0.1464461033 | 0.2204664393, 0.1429954779 (m = 4) |
| REG-22 | prewhitened Bartlett, m = 4 (`neweywest_lag4_prewhite`): p2a fits a scalar AR(1) to the residuals | 0.2858936714, 0.1360607244 | 0.2908704449, 0.3059566437 |
| REG-61 | default bandwidth (`bw_neweywest`; `bw_andrews_qs` 0.6261528081) | `bwNeweyWest` 2.513298742 | fixed rule, 4 |
| REG-61 | `adjust = TRUE`, e.g. Parzen (`kernhac_parzen_bw4_adjust`) | 0.1977403619, 0.1432982597 | no option; 0.1957529711, 0.1418580397 |

Bandwidth bounds since Tier 0 (REG-1, `428d7cf`): a user bandwidth of n or more is an
`InvalidSpecification` error (sandwich truncates the lags with a warning instead; decision D12
keeps the refusal), a user bandwidth above n/4 adds an entry to `HacResult.warnings`, and the
default bandwidth never warns. The case's bandwidth 4 is at most 100/4, so no warning fires.

## Driscoll-Kraay (dk_panel)

DGP (legacy Test 10): `set.seed(42)`, N = 10 firms, T = 20 periods, `x ~ N(0, 1)`, a firm
effect `N(0, 1)` per firm, `y = 2 + 3 x + firm effect + N(0, 1)`; data
`validation/datasets/generated/dk_panel.csv` (columns `y, x, firm, time`). Model: pooled
`lm(y ~ x)`. p2a's default bandwidth is `floor(T^(1/4))` = 2.

R calls: `vcovPL(m, cluster = ~firm, order.by = ~time, lag = 2, adjust = FALSE | TRUE)`
(sandwich 3.1.3) and `vcovSCC(plm(y ~ x, model = "pooling"), maxlag = 2)` (plm 2.6.7) as a
cross-check; both aggregate the scores by period and weight lag j by 1 - j/(lag + 1), which is
p2a's Bartlett weight at bandwidth 2.

| Key | (Intercept) | x |
|---|---|---|
| `coef` | 1.670706922 | 3.039250681 |
| `se_ols` (classical OLS) | 0.1073184461 | 0.1103492879 |
| `vcovpl_lag2` = `vcovscc_lag2` | 0.04889775602 | 0.1013769059 |
| `vcovpl_lag2_adjust` | 0.04914409388 | 0.1018876240 |
| `legacy_vcovpl_cluster_time` (legacy `dk_se`, not a Driscoll-Kraay value) | 0.4115783370 | 0.1331376397 |

`lag` = 2, `n_periods` = 20, `n_obs` = 200.

Test: `test_validate_driscoll_kraay_panel` pins the coefficients, the classical OLS standard
errors (a cross-check of the data), the default bandwidth, the number of periods and n; it also
checks the unit count, the kernel and that no bandwidth warning fires. Tolerance: coefficients
1e-8, standard errors 1e-6 relative, counts exact.

Deviation REG-24: p2a averages the scores within each period and multiplies the meat by T, so
its standard errors, covariance, t and p are sqrt(T)/N = 0.4472135955 times vcovPL's: p2a
0.02186774128, 0.04533713058 against 0.04889775602, 0.1013769059. They are not pinned.

The legacy call `vcovPL(model_pool, cluster = ~time, order.by = ~time)` names the same variable
twice. sandwich then keeps `cluster = time` and derives `order.by` from the occurrence index
within each period, which is the firm here, so it sums the scores by firm with the default lag
`floor(10^(1/4))` = 1. Its result, 0.4115783370 and 0.1331376397, is what
`validation/expected/driscoll_kraay_test.csv` holds as `dk_se`; the case records it as
`legacy_vcovpl_cluster_time` for comparison only.

Bandwidth bounds since Tier 0 (REG-1, `428d7cf`), with T in place of n: a user bandwidth of T
or more is an `InvalidSpecification` error, one above T/4 adds an entry to
`DriscollKraayResult.warnings`, and the default bandwidth never warns.

## Legacy files superseded

| Legacy file (`validation/expected/`) | Superseded by | Note |
|---|---|---|
| `bgtest_test.csv` | `diag_bgtest_ar1` | values equal to 15 digits |
| `reset_test.csv` | `diag_reset_quad` | values equal to 15 digits |
| `harveycollier_test.csv` | `diag_harvtest_nonlin` | values equal to 15 digits |
| `wald_test.csv` | `diag_wald_sim` | `p_value` is `NA`: `aod::wald.test` names its p-value `P`, not `P(> Chi2)`; the case records it |
| `hac_test.csv` | `hac_ar1` | regenerated (INF-16): rows "Bartlett" and "Parzen" held `vcovHAC(m)`'s default quadratic-spectral estimate, because `vcovHAC` ignores a `kernel` argument; they now hold `kernhac_bartlett_bw4` and `kernhac_parzen_bw4`. Row "NeweyWest" equals `neweywest_default` |
| `driscoll_kraay_test.csv` | `dk_panel` | coefficients and `ols_se` equal; `dk_se` sums the scores by firm, not by period (above), and is not a Driscoll-Kraay value |
| `bootstrap_test.csv` | none | not asserted: `vcovBS` resamples with R's RNG (see `validation/regression/vcovBS.md`) |
