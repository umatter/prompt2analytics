# Validation: Instrumental Variables / Two-Stage Least Squares (2SLS)

## Method

`run_iv2sls(dataset, y, x_exog, x_endog, instruments, robust)` estimates
`y = [1, X_exog] b_exog + X_endog b_endog + e` by 2SLS with the instrument matrix
`Z = [1, X_exog, Z_excl]`: `b = (X'P_Z X)^-1 X'P_Z y`, computed from the k x k cross
products. `robust = false` gives the classical covariance `s^2 (X'P_Z X)^-1` with
`s^2 = e'e / (n - k)`; `robust = true` the HC1 sandwich (`n / (n - k)` times HC0), as
`sandwich::vcovHC(type = "HC1")`. `sargan_test` (and `run_sargan_test`) computes Sargan's
`n R^2` of the 2SLS residuals on Z, chi-squared with `L - K` degrees of freedom.
`run_first_stage_diagnostics(dataset, endog, instruments, controls)` fits the first stage
of one endogenous variable on its own.

## What p2a reports

- **First-stage F** (`IVResult::first_stage_f_stats`, one per endogenous variable, in
  `x_endog` order): the partial F of the L excluded instruments, comparing the restricted
  fit `x_j ~ [1, X_exog]` with `x_j ~ [1, X_exog, Z_excl]`, on `(L, n - k_z)` degrees of
  freedom (`first_stage_df`, `k_z` = columns of `[1, X_exog, Z_excl]`). This is AER's
  "Weak instruments" statistic (`summary(ivreg, diagnostics = TRUE)`). With
  `robust = true` it is the HC1 Wald F of the excluded-instrument coefficients,
  `b' V_bb^-1 b / L`, as AER with `vcov. = sandwich::vcovHC(type = "HC1")`;
  `first_stage_f_type` says which.
- `first_stage_p_values`: upper-tail F probabilities (`sf`, not `1 - cdf`).
- `first_stage_partial_r2`: `1 - RSS(full) / RSS(restricted)`, classical in both modes.
- `strong_instruments` is true when every first-stage F reaches 10 (the rule of thumb); a
  smaller (or not computed) F adds the warning "Weak instruments for 'x': first-stage ...
  = F on (L, df2) DF, below 10", and `iv_diagnostics` reports a WEAK_INSTRUMENT warning
  that names the statistic and its df. Stock-Yogo critical values and the Anderson-Rubin
  test are not computed.
- `run_first_stage_diagnostics`: the partial F of the instruments given the controls,
  `anova(lm(x ~ controls), lm(x ~ controls + instruments))`, on `(L, n - k)` with
  `k = 1 + L + n_controls` (`f_df`), its p-value, the partial R-squared and the R-squared
  and adjusted R-squared of the full first stage. Without controls the partial F is the
  overall F of the first stage.

Before Tier 2 the first-stage F was the overall F of `x_j ~ [1, X_exog, Z]` against an
intercept-only model, so a control that predicts the endogenous regressor certified
irrelevant instruments as strong (ECON-3; `iv_weak_controls`: 300.81 where AER's
weak-instrument F is 0.2267), and `robust = true` still reported the classical statistic.

## Reference

AER 1.2.17 `ivreg`, `summary(m, diagnostics = TRUE)` (its `ivdiag` computes the
weak-instrument F as the partial F of the excluded instruments), with
`vcov. = function(x) sandwich::vcovHC(x, type = "HC1")` (sandwich 3.1.3) for the robust F;
the first-stage diagnostics against `anova(lm(...), lm(...))` and `summary(lm(...))`.
R 4.3.3.

## R cases

`validation/r/cases/econ_iv.R`:

| Case | Design | n |
|---|---|---|
| `iv_weak_controls` | `ivreg(y ~ x + w \| w + z1 + z2)`: a control `w` that predicts `x`, weak excluded instruments | 500 |
| `iv_two_endog` | `ivreg(y ~ x1 + x2 + w1 + w2 \| w1 + w2 + z1 + z2 + z3)`: two endogenous regressors | 400 |
| `iv_no_controls` | `ivreg(y ~ x \| z1 + z2)`: no controls (the first-stage F was already right) | 300 |
| `iv_na_rows` | `ivreg(y ~ x + w + k \| w + k + z1 + z2)` on data with missing cells in six columns, an integer control `k` and a text column `label` | 300 (294 used) |

## Pinned values (R, the case files hold 17 digits)

| Case | Quantity | R |
|---|---|---|
| `iv_weak_controls` | weak-instrument F (2, 496), p | 0.22671626635951783, 0.79722948956125039 |
| | HC1 Wald F, p | 0.23166413083315232, 0.79329827271905684 |
| | partial R², first-stage R², adj. R² | 0.0009133435343690044, 0.64531532143409298, 0.64317005120083137 |
| | coef (Intercept), x, w | 1.0425984384578373, 0.10838033928024544, 1.3271390550714173 |
| | SE classical | 0.083341471514729454, 1.8355768531997416, 3.7517822785945838 |
| | SE HC1 | 0.085334611710189454, 1.8351318071923248, 3.7463531763495266 |
| | Sargan, p | 1.4718814653242318, 0.22504907533747728 |
| | first stage z1, z2 coef / SE | -0.0090167616002719778, 0.044734024592271097 / 0.065289478724813818, 0.067078769668775656 |
| | overall first-stage F (the pre-fix value, not compared) | 300.80840699168988 |
| `iv_two_endog` | weak-instrument F x1, x2 (3, 394) | 30.182002003721841, 24.396733081297839 |
| | p | 1.3891011676843923e-17, 1.6837340107205339e-14 |
| | HC1 Wald F, p | 33.410423394721207, 26.602360737529967 / 2.9295517483564005e-19, 1.0921970131748287e-15 |
| | coef (Intercept), x1, x2, w1, w2 | 1.9748588721547791, 0.83573137092603222, -0.72366720119861727, 0.36292478602969902, 0.43062617093990935 |
| | Sargan, p | 0.41282025131827993, 0.52054151700815932 |
| `iv_no_controls` | weak-instrument F (2, 297), p | 8.8616789134912786, 0.00018275941770054888 |
| | HC1 Wald F, p | 9.349761011317689, 0.00011538648832507372 |
| | coef (Intercept), x | 1.1021230930489567, 0.65111102795573639 |
| | Sargan, p | 0.9401159877792109, 0.33224801435443996 |
| `iv_na_rows` | coef (Intercept), x, w, k | 1.0554082366227837, 0.54877689731239732, 0.35628326343265526, -0.3410347236015594 |
| | SE classical | 0.17163940376487344, 0.071434817505935458, 0.07925639250311578, 0.047390729798043416 |
| | SE HC1 | 0.15904354059293663, 0.073322644355988092, 0.073621644706330924, 0.047104803764284893 |
| | weak-instrument F (2, 289), p | 84.915225706067275, 9.7946873967282209e-30 |
| | HC1 Wald F, p | 88.013921635429739, 1.4093829753259807e-30 |
| | Sargan, p | 0.11130482093750826, 0.73866347796076148 |
| | n used, dropped; missing per column | 294, 6; y 2, x 1, w 1, k 1, z1 1, z2 1 |
| | first-stage diagnostics (rows complete in x, z1, z2, w, k) | n 296; partial F 83.119221603270859 on (2, 291), p 2.7903687835782346e-29; partial R² 0.36357057390174263; R² 0.3893239601294386; adj. 0.38092978776008379 |

## Tests

Registered as `iv_2sls` in `validation/methods/econ_iv_panel_na.json`:

- `p2a-core`, `econometrics::iv::tests`: `test_validate_iv_weak_controls`,
  `test_validate_iv_two_endog`, `test_validate_iv_no_controls` (coefficients, classical and
  HC1 SEs, both F statistics with df and p, partial R², Sargan, n);
  `test_validate_first_stage_diagnostics_weak_controls` (partial F given the controls, df,
  p, partial R², R², adjusted R², instrument coefficients and SEs; without controls on
  `iv_no_controls`, equal to the weak-instrument F); `test_validate_iv_display_first_stage`
  (the printed first-stage block); `test_validate_iv_na_rows` (all of the above on the 294
  complete rows, the missing-value counts and the note);
  `test_validate_na_iv_errors_name_the_column` (a text column is `NonNumericColumn`, an
  unknown one `ColumnNotFound`; an Int64 control equals the same values in Float64).
- `p2a-core`, `diagnostics::iv::tests`: `test_validate_iv_diagnostics_weak_controls`
  (Critical WEAK_INSTRUMENT warning naming the partial or HC1 Wald F and its df, for
  classical and robust fits) and `test_validate_iv_diagnostics_two_endog` (no warning;
  Sargan not rejected).
- `p2a-core`, `econometrics::general_gmm::tests::test_validate_na_gmm_iv_omit_equals_complete_rows`:
  `run_gmm_iv` on `iv_na_rows` equals the fit on its complete rows (not pinned to R: the
  GMM variance is ECON-21, Tier 4).
- `p2a-mcp`, `tier2::wp08_iv_panel`: `test_validate_mcp_iv_2sls_weak_controls`,
  `test_validate_mcp_iv_first_stage_weak_controls`, `test_validate_mcp_iv_sargan_na_rows`
  (the tool texts at print precision).

## Tolerance

`validation/README.md`, n = 294-500: coefficients 1e-8 relative, standard errors and
statistics 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; counts and degrees of
freedom exact. The largest differences observed are about 1e-12 relative (the coefficient
on `w` in `iv_weak_controls`, where the instruments are weak).

## Not compared

- `se_hc0`: `run_iv2sls` offers the classical and the HC1 covariance only.
- `wu_hausman`: the Wu-Hausman test is not computed.
- `overall_first_stage_f`: the statistic p2a reported before Tier 2, kept in the case for
  the record.

## Missing values

`run_iv2sls`, `sargan_test` and `run_gmm_iv` drop the rows with a missing value (null or
NaN) in y, a regressor or an instrument, jointly (ivreg `na.omit`), and report them in
`missing` and in the output ("6 of 300 rows dropped because of missing values (y: 2, w: 1,
k: 1, x: 1, z1: 1, z2: 1)"). `run_first_stage_diagnostics` uses the complete rows of its
own columns (the endogenous variable, the instruments and the controls), which can be
more than the 2SLS rows because it does not use y. An infinite value is refused
(`NonFiniteValues`), a text column is `NonNumericColumn`.

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::iv::tests::test_validate diagnostics::iv
cargo test -p p2a-mcp --test tier2 --no-default-features --features "http,llm,auth,websocket" -- wp08_iv_panel
Rscript validation/r/run.R --check iv_weak_controls iv_two_endog iv_no_controls iv_na_rows
```

## References

- Angrist, J.D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton
  University Press. Chapter 4.
- Kleiber, C. & Zeileis, A. (2008). *Applied Econometrics with R*. Springer (AER).
- Sargan, J.D. (1958). The estimation of economic relationships using instrumental
  variables. *Econometrica*, 26(3), 393-415.
- Staiger, D. & Stock, J.H. (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3), 557-586.
- Stock, J.H. & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In
  *Identification and Inference for Econometric Models*. Cambridge University Press.
