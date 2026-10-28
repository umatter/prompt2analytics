# Validation: Panel Fixed Effects (within)

## Method and options

The within estimator demeans `y` and the regressors by entity and runs OLS on the demeaned
data, which removes every time-invariant entity effect:

```
y_it - ybar_i = (x_it - xbar_i)' beta + (e_it - ebar_i)
```

- Core: `p2a_core::run_fixed_effects(dataset, y, x, entity)` (classical covariance) and
  `p2a_core::run_fixed_effects_with(dataset, y, x, entity, &PanelConfig { vcov })`.
- MCP tool: `panel_fixed_effects`, with the optional `vcov` argument.
- `vcov` (`PanelVcov`): `classical` (default, as plm: `s^2 (X~'X~)^-1` with
  `s^2 = SSR / (n - N - K)`), `cluster` (entity-clustered Arellano sandwich with Stata's
  small-sample factor `G/(G-1) (n-1)/(n-K)`, plm `type = "sss"`), `cluster-hc0` (factor 1),
  `cluster-hc1` (factor `n/(n-K)`). `K` is the number of slopes, `G = N` the number of
  entities.
- Inference: Student t with `df = n - N - K`. R-squared is the within R-squared; the adjusted
  within R-squared is plm's `r.squared(dfcor = TRUE)`, `1 - (1 - R2)(n - 1)/df`. The F test
  of the slopes follows `vcov`, as plm's `summary(fe, vcov = ...)`: the classical within F
  test on `(K, n - N - K)` df, or with a clustered `vcov` the Wald F `W/K` on `(K, G - 1)` df
  (plm `pwaldtest(test = "F")` with a cluster vcov, its `df2adj`). `PanelResult::f_df`
  returns the two degrees of freedom.
- Refused: a non-finite value in `y` or a regressor, and regressors that are exactly
  collinear after removing the entity means (also a regressor that is exactly zero after
  demeaning), whose coefficients are not identified.

## R reference

Cases `panel_fe_grunfeld`, `panel_fe_grunfeld_unbalanced` and `panel_fe_sim_corr`, defined in
`validation/r/cases/panel.R` and written to `validation/expected/<id>.json` by
`Rscript validation/r/run.R --write`. R 4.3.3, plm 2.6.7:

```r
pd <- pdata.frame(d, index = c("firm", "year"))
fe <- plm(inv ~ value + capital, data = pd, model = "within")
summary(fe)$r.squared; summary(fe)$fstatistic
vcovHC(fe, method = "arellano", type = "HC0")   # also "HC1", "sss"
# robust F, = summary(fe, vcov = ...)$fstatistic, on (K, G - 1) df
pwaldtest(fe, test = "F", vcov = vcovHC(fe, method = "arellano", type = "sss"))
```

The datasets are `validation/datasets/grunfeld.csv` and the generated
`validation/datasets/generated/grunfeld_unbalanced.csv` and
`validation/datasets/generated/panel_corr_effects.csv`, which R writes and reads back
before computing, so R and Rust see the same doubles.

## Pinned values (R, 12 significant digits; the case files hold 17)

#### `panel_fe_grunfeld`: Grunfeld (balanced, N = 10, T = 20, n = 200)

| quantity | `value` | `capital` |
|---|---|---|
| coefficient | 0.110123804121 | 0.3100653413 |
| SE (classical) | 0.011856694214 | 0.0173545027756 |
| t | 9.28790117487 | 17.8665643902 |
| p | 3.92110843164e-17 | 2.22000669284e-42 |
| 95% CI lower | 0.0867345457897 | 0.27583076113 |
| 95% CI upper | 0.133513062452 | 0.34429992147 |
| SE Arellano HC0 | 0.0143421437124 | 0.0497926087238 |
| SE Arellano HC1 | 0.0144143967828 | 0.0500434546878 |
| SE Arellano sss | 0.0151560754389 | 0.0526183915915 |

| quantity | value |
|---|---|
| within R² | 0.766757583748 |
| adj. within R² | 0.753110421095 |
| F | 309.014175168 |
| F df1 | 2 |
| F df2 | 188 |
| F p-value | 3.74893568134e-60 |
| F Arellano HC0 (Wald W/K) | 31.7744320126 |
| F Arellano HC0 p-value | 8.3416798482e-05 |
| F Arellano HC1 (Wald W/K) | 31.4566876925 |
| F Arellano HC1 p-value | 8.67856181418e-05 |
| F Arellano sss (Wald W/K) | 28.45328535 |
| F Arellano sss p-value | 0.000128502647786 |
| F Arellano df2 (G - 1) | 9 |
| σ²_e | 2784.45823078 |
| df_resid | 188 |
| n | 200 |
| N | 10 |

#### `panel_fe_grunfeld_unbalanced`: Grunfeld without the rows where `(firm + year) %% 7 == 0` (n = 171)

| quantity | `value` | `capital` |
|---|---|---|
| coefficient | 0.0994514002277 | 0.321189365789 |
| SE (classical) | 0.013732594315 | 0.0192878456077 |
| t | 7.24199651911 | 16.6524231022 |
| p | 1.79102212411e-11 | 1.11003372717e-36 |
| 95% CI lower | 0.072329578464 | 0.283095943762 |
| 95% CI upper | 0.126573221991 | 0.359282787817 |
| SE Arellano HC0 | 0.0184394267895 | 0.0517539659599 |
| SE Arellano HC1 | 0.018548214912 | 0.0520593017415 |
| SE Arellano sss | 0.0194942831613 | 0.0547146328711 |

| quantity | value |
|---|---|
| within R² | 0.755418407833 |
| adj. within R² | 0.738497668752 |
| F | 245.544903403 |
| F df1 | 2 |
| F df2 | 159 |
| F p-value | 2.39711871119e-49 |
| F Arellano HC0 (Wald W/K) | 20.3123217299 |
| F Arellano HC0 p-value | 0.000460736898814 |
| F Arellano HC1 (Wald W/K) | 20.0747507155 |
| F Arellano HC1 p-value | 0.000481122056558 |
| F Arellano sss (Wald W/K) | 18.173553736 |
| F Arellano sss p-value | 0.000691222929933 |
| F Arellano df2 (G - 1) | 9 |
| σ²_e | 3025.93594284 |
| df_resid | 159 |
| n | 171 |
| N | 10 |

#### `panel_fe_sim_corr`: simulated, effects correlated with `value` (N = 60, T = 5, n = 300)

| quantity | `value` | `capital` |
|---|---|---|
| coefficient | 0.935862128154 | -0.500031221073 |
| SE (classical) | 0.0651754894431 | 0.0597049747268 |
| t | 14.3591116254 | -8.37503446507 |
| p | 4.27018079768e-34 | 4.78718198014e-15 |
| 95% CI lower | 0.807467617773 | -0.617648918506 |
| 95% CI upper | 1.06425663853 | -0.38241352364 |
| SE Arellano HC0 | 0.0677702088159 | 0.0615308188017 |
| SE Arellano HC1 | 0.0679972453273 | 0.0617369527754 |
| SE Arellano sss | 0.0684566914004 | 0.0621540990933 |

| quantity | value |
|---|---|
| within R² | 0.519423852056 |
| adj. within R² | 0.396250973801 |
| F | 128.61944701 |
| F df1 | 2 |
| F df2 | 238 |
| F p-value | 1.34804139647e-38 |
| F Arellano HC0 (Wald W/K) | 109.077862607 |
| F Arellano HC0 p-value | 1.51279695931e-20 |
| F Arellano HC1 (Wald W/K) | 108.350676856 |
| F Arellano HC1 p-value | 1.7667977475e-20 |
| F Arellano sss (Wald W/K) | 106.901169474 |
| F Arellano sss p-value | 2.41332438877e-20 |
| F Arellano df2 (G - 1) | 59 |
| σ²_e | 0.877888572173 |
| df_resid | 238 |
| n | 300 |
| N | 60 |


## Tests

- `p2a-core`: `econometrics::panel::tests::test_validate_panel_fe_grunfeld`,
  `econometrics::panel::tests::test_validate_panel_fe_grunfeld_unbalanced`,
  `econometrics::panel::tests::test_validate_panel_fe_sim_corr` pin every quantity above,
  the classical and the three Arellano standard errors and F tests.
- Unit tests: `econometrics::panel::linear_models::tests::test_fixed_effects_refuses_non_finite_data`,
  `econometrics::panel::tests::test_collinear_regressors_are_refused`,
  `econometrics::panel::tests::test_time_invariant_regressor_fe_refuses_re_warns`.
- `p2a-mcp`: `tools::handlers::panel::tests::test_validate_mcp_panel_fixed_effects` checks
  the `panel_fixed_effects` text against the case at display precision, with the default
  and with `vcov = "cluster-hc0"`.
- Registry: `validation/methods/panel.json`, method `panel_fe`.

## Tolerance

`validation/README.md`, n = 120-300: coefficients 1e-8 relative, standard errors and
statistics 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; counts and degrees of
freedom exact.

## What changed in Tier 1

- Adjusted R-squared: p2a used `(n - N - 1)/df` instead of plm's `(n - 1)/df`. Grunfeld
  0.765517 (was) against plm 0.753110; unbalanced 0.753880 against 0.738498; simulated
  0.517405 against 0.396251 (ECON-73).
- The entity-clustered (Arellano) option is new; with it the F test is plm's robust Wald F.
- Non-finite data and exactly collinear regressors are refused. A NaN in `y` gave NaN
  coefficients with SE 0 and p 1; collinear regressors gave the minimum-norm split.
- The earlier tests used a hand-typed five-firm subset and a full-Grunfeld test at 0.01
  tolerance; both are replaced by the case-pinned tests above. Coefficients, classical
  standard errors, within R-squared and F were already equal to plm.

## Known differences

None. `panel_fe_synthetic.csv` in `validation/expected/` is a legacy file that no test asserts.

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::panel::tests::test_validate_panel_fe
cargo test -p p2a-mcp --lib --no-default-features --features "http,llm,auth,websocket" -- tools::handlers::panel::tests
Rscript validation/r/run.R --check
```

## References

- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*, 5th ed. Wiley. Chapter 2.
- Arellano, M. (1987). Computing robust standard errors for within-groups estimators.
  *Oxford Bulletin of Economics and Statistics*, 49(4), 431-434.
- Croissant, Y. & Millo, G. (2008). Panel data econometrics in R: The plm package.
  *Journal of Statistical Software*, 27(2), 1-43.
