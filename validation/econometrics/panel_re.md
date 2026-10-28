# Validation: Panel Random Effects (Swamy-Arora GLS)

## Method and options

Random effects treats the entity effects as random, uncorrelated with the regressors, and
estimates by GLS through quasi-demeaning with one factor per entity:

```
y_it - theta_i ybar_i = (1 - theta_i) alpha + (x_it - theta_i xbar_i)' beta + error
theta_i = 1 - sqrt(sigma2_nu / (T_i sigma2_eta + sigma2_nu))
```

- Core: `p2a_core::run_random_effects(dataset, y, x, entity)` (classical covariance) and
  `p2a_core::run_random_effects_with(dataset, y, x, entity, &PanelConfig { vcov })`.
- MCP tool: `panel_random_effects`, with the optional `vcov` argument.
- Variance components (plm `ercomp(method = "swar")`, one code path for balanced and
  unbalanced panels): `sigma2_nu = SSR_within / (n - N - K)`; the between regression on the
  observation-level group means (entity `i` weighted by `T_i`) gives `quad2` and the
  Baltagi-Chang trace term `M_eta = n - tr[(sum T_i zbar zbar')^-1 sum T_i^2 zbar zbar']`;
  `sigma2_eta = (quad2 - (N - r) sigma2_nu) / M_eta` with `r` the rank of the between design
  (K + 1 unless a regressor does not vary between entities). A negative `sigma2_eta` is set
  to 0 with a warning, so RE becomes pooled OLS, as plm.
- `vcov` (`PanelVcov`): `classical` (default, plm's `vcov`: `s^2 (X*'X*)^-1` with `s^2` the
  residual variance of the transformed model, `n - K - 1` df), `cluster` (Arellano
  sandwich with factor `G/(G-1) (n-1)/(n-K-1)`, plm `type = "sss"`), `cluster-hc0` (factor 1),
  `cluster-hc1` (factor `n/(n-K-1)`).
- Inference: normal (z), as plm's `summary` for random models. R-squared is plm's
  `cor(y*, X*b)^2` of the transformed model; adjusted `1 - (1 - R2)(n - 1)/(n - K - 1)`. The
  joint test of the slopes is the Wald chi-squared `b_s' V_ss^-1 b_s` with the chosen
  covariance (`wald_chi2`, `wald_df`, `wald_p_value`); `f_stat` is its F form `W/K` on
  `(K, n - K - 1)` df (plm `pwaldtest(test = "F")`).
- `theta` is the observation-weighted mean `sum T_i theta_i / n` (plm's mean theta);
  `theta_by_group` holds the `theta_i` on unbalanced panels.
- Refused: `n <= N + K` (no within degrees of freedom) and `N <= K + 1` (the between
  regression is not estimable, as plm).

## R reference

Cases `panel_re_grunfeld`, `panel_re_grunfeld_unbalanced`, `panel_re_sim_corr` and
`panel_re_sim_no_effect`, defined in `validation/r/cases/panel.R`. R 4.3.3, plm 2.6.7:

```r
pd <- pdata.frame(d, index = c("firm", "year"))
re <- plm(inv ~ value + capital, data = pd, model = "random")
sqrt(diag(vcov(re))); ercomp(re); summary(re)$r.squared
summary(re)$fstatistic            # Wald chi-squared for random models
pwaldtest(re, test = "F")         # its F form W/K
vcovHC(re, method = "arellano", type = "HC0")   # also "HC1", "sss"
```

`panel_re_sim_no_effect` (seed 3, N = 30, T = 4, no entity effect) has a negative Swamy-Arora
`sigma2_eta` (-0.0440); plm sets it to 0, so theta = 0 and the coefficients equal pooled
`lm()` (`coef_pooled_ols`). Datasets: `validation/datasets/grunfeld.csv`,
`validation/datasets/generated/grunfeld_unbalanced.csv`,
`validation/datasets/generated/panel_corr_effects.csv`,
`validation/datasets/generated/panel_no_effect.csv`.

## Pinned values (R, 12 significant digits; the case files hold 17)

#### `panel_re_grunfeld`: Grunfeld (balanced, n = 200)

| quantity | `(Intercept)` | `value` | `capital` |
|---|---|---|---|
| coefficient | -57.834414905 | 0.109781152232 | 0.308112982831 |
| SE (classical) | 28.8989352603 | 0.0104926635495 | 0.0171804690896 |
| z | -2.00126455816 | 10.462658191 | 17.9339097916 |
| p (normal) | 0.0453638870272 | 1.28207497963e-25 | 6.4108791184e-72 |
| 95% CI lower | -114.475287207 | 0.0892159095735 | 0.274439882178 |
| 95% CI upper | -1.19354260331 | 0.130346394891 | 0.341786083484 |
| SE Arellano HC0 | 23.4496261098 | 0.0129840196125 | 0.0518890249063 |
| SE Arellano HC1 | 23.6275019289 | 0.0130825091625 | 0.0522826261844 |
| SE Arellano sss | 24.8432318787 | 0.0137556568468 | 0.0549727774624 |

| quantity | value |
|---|---|
| σ²_ν (idiosyncratic) | 2784.45823078 |
| σ²_η (individual) | 7089.80009931 |
| θ mean | 0.861223620748 |
| θ min | 0.861223620748 |
| θ max | 0.861223620748 |
| transformed residual variance | 2786.31500117 |
| R² (cor(y*, X*b)²) | 0.76950272267 |
| adj. R² | 0.767162648788 |
| Wald χ² | 657.673869826 |
| Wald df | 2 |
| Wald p | 1.54146518865e-143 |
| Wald F (W/K) | 328.836934913 |
| Wald F df2 | 197 |
| Wald F p | 1.66948022983e-63 |
| df_resid | 197 |
| n | 200 |
| N | 10 |

#### `panel_re_grunfeld_unbalanced`: Grunfeld without the rows where `(firm + year) %% 7 == 0` (n = 171)

| quantity | `(Intercept)` | `value` | `capital` |
|---|---|---|---|
| coefficient | -51.8073181707 | 0.101715228061 | 0.317775534793 |
| SE (classical) | 29.1313827101 | 0.0116836482271 | 0.0190449203769 |
| z | -1.77840230539 | 8.70577631955 | 16.6855795931 |
| p (normal) | 0.0753378043308 | 3.15408340084e-18 | 1.66878570305e-62 |
| 95% CI lower | -108.903779102 | 0.0788156983281 | 0.280448176766 |
| 95% CI upper | 5.28914276084 | 0.124614757794 | 0.35510289282 |
| SE Arellano HC0 | 25.2107950635 | 0.0148896146832 | 0.0547510667462 |
| SE Arellano HC1 | 25.4348954268 | 0.0150219694166 | 0.0552377524663 |
| SE Arellano sss | 26.7322249596 | 0.0157881783684 | 0.0580552033107 |

| quantity | value |
|---|---|
| σ²_ν (idiosyncratic) | 3025.93594284 |
| σ²_η (individual) | 6907.90235751 |
| θ mean | 0.84196650807 |
| θ min | 0.841507783859 |
| θ max | 0.845865663863 |
| transformed residual variance | 3031.14309248 |
| R² (cor(y*, X*b)²) | 0.759850044445 |
| adj. R² | 0.756991116403 |
| Wald χ² | 531.461227844 |
| Wald df | 2 |
| Wald p | 3.93242728654e-116 |
| Wald F (W/K) | 265.730613922 |
| Wald F df2 | 168 |
| Wald F p | 9.24343534533e-53 |
| df_resid | 168 |
| n | 171 |
| N | 10 |

#### `panel_re_sim_corr`: simulated, effects correlated with `value` (n = 300)

| quantity | `(Intercept)` | `value` | `capital` |
|---|---|---|---|
| coefficient | 1.14501624586 | 1.26295673575 | -0.515856191502 |
| SE (classical) | 0.120563577012 | 0.06506158108 | 0.0666771404906 |
| z | 9.49719869159 | 19.4117129462 | -7.73662739144 |
| p (normal) | 2.15612376559e-21 | 6.14441547716e-84 | 1.02088687578e-14 |
| 95% CI lower | 0.908715977064 | 1.13543838006 | -0.646540985456 |
| 95% CI upper | 1.38131651465 | 1.39047509145 | -0.385171397548 |
| SE Arellano HC0 | 0.152547743517 | 0.0525559655458 | 0.0635978985142 |
| SE Arellano HC1 | 0.153316250867 | 0.052820732791 | 0.0639182929778 |
| SE Arellano sss | 0.154352183265 | 0.053177633694 | 0.0643501782485 |

| quantity | value |
|---|---|
| σ²_ν (idiosyncratic) | 0.877888572173 |
| σ²_η (individual) | 0.473219922241 |
| θ mean | 0.479788359372 |
| θ min | 0.479788359372 |
| θ max | 0.479788359372 |
| transformed residual variance | 1.16929011269 |
| R² (cor(y*, X*b)²) | 0.583388149406 |
| adj. R² | 0.580582682398 |
| Wald χ² | 415.89378729 |
| Wald df | 2 |
| Wald p | 4.89566352149e-91 |
| Wald F (W/K) | 207.946893645 |
| Wald F df2 | 297 |
| Wald F p | 3.38957678108e-57 |
| df_resid | 297 |
| n | 300 |
| N | 60 |

#### `panel_re_sim_no_effect`: simulated without entity effects (N = 30, T = 4, n = 120)

| quantity | `(Intercept)` | `value` | `capital` |
|---|---|---|---|
| coefficient | 2.13336877273 | 0.458470318169 | -0.280439231763 |
| SE (classical) | 0.094894692677 | 0.113589229333 | 0.0843803499637 |
| z | 22.4814340249 | 4.03621294785 | -3.32351349436 |
| p (normal) | 6.30688857231e-112 | 5.4320915354e-05 | 0.000888911219744 |
| 95% CI lower | 1.94737859276 | 0.235839519645 | -0.445821678695 |
| 95% CI upper | 2.3193589527 | 0.681101116693 | -0.115056784831 |
| SE Arellano HC0 | 0.0875401401473 | 0.117614222712 | 0.0724306601942 |
| SE Arellano HC1 | 0.0886553461272 | 0.119112553469 | 0.0733533809626 |
| SE Arellano sss | 0.089794433686 | 0.120642970231 | 0.0742958613352 |
| pooled OLS coefficient | 2.13336877273 | 0.458470318169 | -0.280439231763 |

| quantity | value |
|---|---|
| σ²_ν (idiosyncratic) | 1.12064792061 |
| σ²_η (individual) | 0 |
| θ mean | 0 |
| θ min | 0 |
| θ max | 0 |
| transformed residual variance | 1.07999985332 |
| R² (cor(y*, X*b)²) | 0.201441533614 |
| adj. R² | 0.187790961539 |
| Wald χ² | 29.5140060809 |
| Wald df | 2 |
| Wald p | 3.90045265007e-07 |
| Wald F (W/K) | 14.7570030404 |
| Wald F df2 | 117 |
| Wald F p | 1.92727208253e-06 |
| df_resid | 117 |
| n | 120 |
| N | 30 |


Not compared: `r2_overall_cor`, Stata's "overall" R-squared `cor(y, X b)^2` on the
untransformed data, which p2a reported as `r_squared` before Tier 1.

## Tests

- `p2a-core`: `econometrics::panel::tests::test_validate_panel_re_grunfeld`,
  `econometrics::panel::tests::test_validate_panel_re_grunfeld_unbalanced`,
  `econometrics::panel::tests::test_validate_panel_re_sim_corr`,
  `econometrics::panel::tests::test_validate_panel_re_sim_no_effect` pin every quantity
  above, the classical and the three Arellano standard errors, and check that the Wald
  statistic under the clustered covariance is `b_s' V_ss^-1 b_s` of that covariance (the
  cases hold no robust Wald).
- Unit tests: `econometrics::panel::tests::test_random_effects_between_rank_deficient`,
  `econometrics::panel::linear_models::tests::test_random_effects_needs_within_degrees_of_freedom`,
  `econometrics::panel::linear_models::tests::test_random_effects_needs_more_entities_than_coefficients`,
  `econometrics::panel::linear_models::tests::test_random_effects_stops_when_cancelled`.
- `p2a-mcp`: `tools::handlers::panel::tests::test_validate_mcp_panel_random_effects` checks
  the `panel_random_effects` text against the case at display precision, with the default
  and with `vcov = "cluster"`.
- Registry: `validation/methods/panel.json`, method `panel_re`.

## Tolerance

`validation/README.md`, n = 120-300: coefficients 1e-8 relative, standard errors and
statistics 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; counts and degrees of
freedom exact.

## What changed in Tier 1

- Covariance (ECON-1): p2a reported an entity-cluster sandwich whose meat multiplied the
  quasi-demeaned regressors by the untransformed residuals. Grunfeld SEs were 168.547080,
  0.0335132, 0.0582570 against plm's 28.898935, 0.0104927, 0.0171805; with t(197) p-values
  0.7319, 0.0012, 0.0000.
- Unbalanced panels (ECON-75): the unweighted between regression with harmonic-mean T gave
  coefficients -51.80460, 0.1017104, 0.3177845 (plm -51.80732, 0.1017152, 0.3177755),
  sigma_u^2 6927.96 (plm 6907.90) and one theta 0.8421609 (plm mean 0.8419665, range
  0.8415078-0.8458657).
- Fit statistics (ECON-73): R-squared was Stata's overall R-squared 0.806104 (plm 0.769503),
  adjusted 0.804136 (plm 0.767163), and `f_stat` an F of the untransformed fit, 402.2732
  (plm Wald chi-squared 657.6739 on 2 df, F form 328.8369).
- Small panels: `n <= N + K` fell back to pooled OLS; it is now refused.

## Known differences

- A between regression without full rank (a regressor without between variation) uses a
  generalized inverse with a warning. On balanced panels this equals plm, which drops the
  aliased column; on unbalanced panels plm stops (`dgesv: system is exactly singular`) and
  p2a still returns an estimate.

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::panel::tests::test_validate_panel_re
cargo test -p p2a-mcp --lib --no-default-features --features "http,llm,auth,websocket" -- tools::handlers::panel::tests
Rscript validation/r/run.R --check
```

## References

- Swamy, P.A.V.B. & Arora, S.S. (1972). The exact finite sample properties of the
  estimators of coefficients in the error components regression models. *Econometrica*,
  40(2), 261-275.
- Baltagi, B.H. & Chang, Y.-J. (1994). Incomplete panels: A comparative study of
  alternative estimators for the unbalanced one-way error component regression model.
  *Journal of Econometrics*, 62(2), 67-89.
- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*, 5th ed. Wiley. Chapters 2 and 9.
