# Validation: Arellano-Bond difference GMM and Blundell-Bond system GMM

## Method Overview

`run_gmm` and `run_arellano_bond` (`crates/p2a-core/src/econometrics/panel/dynamic_panel.rs`)
estimate the dynamic panel model

```text
y_it = a_1 y_i,t-1 + ... + a_p y_i,t-p + x_it' b + eta_i + e_it
```

by difference GMM, as R's `plm::pgmm(transformation = "d")`: first differences remove
eta_i, the lagged levels of y instrument the differenced equation, and the strictly
exogenous x enter in differences as their own instruments. `run_arellano_bond` is the
default fit (one lag, two-step, Windmeijer-corrected SEs). The MCP tool `panel_gmm` and
`p2a panel gmm` call `run_gmm`.

Options: `lags` (lags of y among the regressors), `step` (one-step or two-step),
`min_lag`/`max_lag` (instrument lags; default 2 and all available), `collapse` (one
instrument column per lag depth), `robust` (one-step: vcovHC robust instead of
Arellano-Bond (1991) classical; two-step: Windmeijer instead of uncorrected).
`transform = System` runs Blundell-Bond system GMM as `plm::pgmm(transformation = "ld")`;
see [System GMM](#system-gmm-blundell-bond-1998).

## Conventions (those of pgmm)

- **Equations.** The differenced equation of (entity, t) exists only where y is observed
  at t, t-1, ..., t-lags-1, and t is at least max(lags + 1, min_lag) periods after the
  first period of the data. Time runs in integer steps of 1, as plm's time-aware `lag()`
  and Stata without `delta()`: a biennial or date-valued time column forms no equation and
  the error says to recode it as consecutive period numbers. Nothing is zero-filled: gaps
  and late entry drop exactly the equations that need a missing period. A repeated
  (entity, time) row is an error. `n_obs` is the number of equations; `n_groups` the
  number of entities with at least one.
- **Instruments.** For the equation of period t, the entity's observed levels y_s with s
  in [t - max_lag, t - min_lag]: one column per (t, s), block-diagonal by period, or one
  per lag depth t - s with `collapse`. Then one column per differenced x. A column that
  no equation observes is not created.
- **Weights.** One-step: (sum_i Z_i' H_i Z_i)^-1, H tridiagonal with 2 on the diagonal and
  -1 between equations one period apart. Two-step: (sum_i Z_i' e_i e_i' Z_i)^-1 from the
  one-step residuals. A weighting matrix is inverted when, scaled to unit diagonal, its
  smallest eigenvalue exceeds L eps times its largest (L instruments); otherwise it gets
  `MASS::ginv`'s generalized inverse (eigenvalues with |lambda| <= sqrt(eps) max |lambda|
  dropped), as pgmm, and a warning naming the rank. The scaling makes the decision
  independent of the units of y and x.
- **Standard errors.** One-step: Arellano-Bond (1991) classical sigma2 (X'Z A Z'X)^-1 with
  sigma2 = sum e^2 / (2 (n_obs - k)) from the differenced one-step residuals (Stata
  `xtabond`), or robust as `plm::vcovHC.pgmm`. Two-step: uncorrected (X'Z A Z'X)^-1, or the
  Windmeijer (2005) correction of `vcovHC.pgmm`. `se_type` names the type.
- **Tests.** Sargan/Hansen as `plm::sargan` (two-step weights, for one- and two-step fits),
  df = instruments - parameters. One-step fits also report the Arellano-Bond (1991) Sargan
  statistic under homoskedastic errors, (Z'e)' (sum_i Z_i' H Z_i)^-1 (Z'e) / sigma2. The
  AR(1)/AR(2) m-tests as `plm::mtest` with the robust covariance (one-step vcovHC, two-step
  Windmeijer); an m-test with no pair of equations that many periods apart is NaN with a
  warning.
- **Missing values.** Rows with a missing value in y, x, the entity or the time are dropped
  (`na.omit`) and reported in `missing`, `warnings` and the last line of the output.
  Entity ids may be text or numbers; time must hold whole numbers or dates.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pgmm()`, `vcovHC.pgmm()`, `sargan()`, `mtest()` | 2.6.7 (R 4.3.3) |
| MASS | R | `ginv()` (inside pgmm; ranks in `dpd_ab_grunfeld`) | 7.3.60.0.1 |

**Status**: R-pinned (registry entry `panel_gmm_difference` in
`validation/methods/econ_dynamic_panel.json`).

## R reference cases

`validation/r/cases/econ_dynamic_panel.R` ([source](../r/cases/econ_dynamic_panel.R)). Every
case fits `pgmm(..., effect = "individual", transformation = "d")` on the complete rows,
one-step and two-step, and reads the SEs from `summary(robust = FALSE)` and
`summary(robust = TRUE)`. The one-step classical SEs and sigma2 are computed in the case
from pgmm's residuals and `B1` (pgmm's own non-robust one-step matrix omits sigma2), as is
the homoskedastic Sargan statistic. The instrument count and Sargan df are pgmm's non-zero
columns (`*_n_instruments_nonzero`, `*_sargan_df_nonzero`); pgmm's own count is recorded as
`*_n_instruments`.

### `dpd_ab_balanced`: balanced, N = 100, T = 8

`validation/datasets/generated/ab_balanced.csv`: y_t = 0.5 y_t-1 + 0.3 x_t + a_i + e_t,
`y ~ lag(y, 1) + x | lag(y, 2:99)`; names in R's spelling (`lag(y, 1)`, `x`). Variants:
two lags (`lag(y, 1:2)`), instrument lags 2..3 (`lag(y, 2:3)`, max_lag 3), lags 3 and
deeper (`lag(y, 3:99)`, min_lag 3), collapsed instruments, and no x.

| Quantity | `onestep` | `twostep` |
|---|---|---|
| coefficients | 0.42357724570337607, 0.29401392447531266 | 0.37752052312353984, 0.2874922911086821 |
| SE one-step Arellano-Bond (1991) | 0.09442034947963217, 0.04962759667173084 |  |
| SE one-step robust (vcovHC) | 0.0942981259048615, 0.04269139026631887 |  |
| sigma2 | 0.9663412013766516 |  |
| Sargan homoskedastic (p) | 28.255017323308362 (0.10349869119820346) |  |
| SE two-step uncorrected |  | 0.0821088529060171, 0.03811284529461422 |
| SE two-step Windmeijer |  | 0.11587057125880933, 0.049620941774623645 |
| Sargan/Hansen (df, p) | 26.914015871078984 (20, 0.1377004736700713) | 26.593175694681666 (20, 0.1471025143073501) |
| AR(1) z (p) | -5.373505400091766 (7.722049158372895e-08) | -4.323985219679866 (1.53235567211357e-05) |
| AR(2) z (p) | -0.35203737769968224 (0.7248102311026187) | -0.46578069929982235 (0.6413724816805958) |
| equations | 600 | 600 |
| instruments (non-zero; pgmm) | 22; 22 | 22; 22 |

| Quantity | `lags2_twostep` | `maxlag3_twostep` | `collapse_twostep` |
|---|---|---|---|
| coefficients | 0.3430409805817764, -0.04773132172037459, 0.3040184359999837 | 0.48195043153166306, 0.3126038138077341 | 0.6282738667147281, 0.30153822426959087 |
| SE two-step uncorrected | 0.12346799363984286, 0.05435506364537461, 0.04329160923145475 | 0.11138246898962184, 0.042392712830918354 | 0.13966028256851964, 0.045831035443107786 |
| SE two-step Windmeijer | 0.17664374555350287, 0.06873131100299505, 0.05795470905397676 | 0.1420705938092302, 0.050686308063309635 | 0.16132096576629373, 0.04966851522306797 |
| Sargan/Hansen (df, p) | 25.124792778871825 (18, 0.12151716596991052) | 15.22169259088814 (10, 0.12418595649377008) | 6.660290281888169 (5, 0.24715529756484908) |
| AR(1) z (p) | -3.3324759110619646 (0.000860769217865005) | -4.554285059150034 (5.256403841579915e-06) | -4.8345635153160025 (1.3343809589308643e-06) |
| AR(2) z (p) | 0.8258216076584259 (0.40890529492990524) | -0.24388646258180752 (0.8073187539553135) | -0.07206820394189861 (0.9425476299848919) |
| equations | 500 | 600 | 600 |
| instruments (non-zero; pgmm) | 21; 21 | 12; 12 | 7; 7 |

| Quantity | `minlag3_onestep` | `nox_onestep` |
|---|---|---|
| coefficients | 0.33261933168283253, 0.2849821103531628 | 0.4003387484513957 |
| SE one-step Arellano-Bond (1991) | 0.16025910723080142, 0.056987193390147343 | 0.09680726689597186 |
| SE one-step robust (vcovHC) | 0.18570238968221733, 0.053045619932206294 | 0.10298298386530097 |
| sigma2 | 0.9089405877870463 | 1.020235922867029 |
| Sargan homoskedastic (p) | 15.850591774730717 (0.3225834290247436) | 36.66024292339417 (0.012848857509552707) |
| Sargan/Hansen (df, p) | 18.215927665390943 (14, 0.1971226062330471) | 30.62139477583048 (20, 0.06039121779806516) |
| AR(1) z (p) | -2.639054673301248 (0.00831375717631191) | -5.235585432602514 (1.6446272894641167e-07) |
| AR(2) z (p) | 0.20049045528218037 (0.8410970220397003) | -0.616400253289504 (0.5376303862262559) |
| equations | 500 | 600 |
| instruments (non-zero; pgmm) | 16; 16 | 21; 21 |

### `dpd_ab_gaps`: interior gaps and late entry

`validation/datasets/generated/ab_gaps.csv`: the balanced panel without 2004 for ids
divisible by 9 and without 2000-2001 for ids divisible by 13, so 553 equations instead of
600 (id 9 has equations at 2002, 2003 and 2007 only).

| Quantity | `onestep` | `twostep` | `lags2_twostep` | `collapse_twostep` |
|---|---|---|---|---|
| coefficients | 0.4186843810911825, 0.3127265246557439 | 0.4026276956406289, 0.3071181050829104 | 0.3485194232687942, -0.013836131302996928, 0.3191326428499455 | 0.6768018842112927, 0.34017640164286544 |
| SE one-step Arellano-Bond (1991) | 0.09728593786287401, 0.0521747799546413 |  |  |  |
| SE one-step robust (vcovHC) | 0.0992544303312403, 0.04526894902156003 |  |  |  |
| sigma2 | 0.9746570158120493 |  |  |  |
| Sargan homoskedastic (p) | 31.053534186450197 (0.05448767223850292) |  |  |  |
| SE two-step uncorrected |  | 0.08808885185522236, 0.03916277601374306 | 0.11988697630405709, 0.0543812710545726, 0.043804862694906625 | 0.139632210596111, 0.049687659143879855 |
| SE two-step Windmeijer |  | 0.13098078602632593, 0.052445619371485355 | 0.17482919861499402, 0.0747972673415585, 0.05681273153054465 | 0.1494624559931416, 0.05214995798078571 |
| Sargan/Hansen (df, p) | 30.330211331726847 (20, 0.06467683101543238) | 30.284528229247318 (20, 0.06537260196227529) | 23.22525970886267 (18, 0.1820816471829538) | 4.537553887753491 (5, 0.47487463487671616) |
| AR(1) z (p) | -5.224224962372557 (1.748860585673091e-07) | -4.069097392518375 (4.719561743743458e-05) | -3.2743156453376834 (0.0010591822615122616) | -5.338140744038472 (9.390455356858315e-08) |
| AR(2) z (p) | 0.032978512675773855 (0.9736917227202851) | 0.0026359051497193533 (0.9978968544127396) | 0.7587938694078724 (0.4479758735413123) | 0.29678780226159884 (0.766628518513695) |
| equations | 553 | 553 | 442 | 553 |
| instruments (non-zero; pgmm) | 22; 22 | 22; 22 | 21; 21 | 7; 7 |

### `dpd_ab_common_gap`: a year no unit observes, missing cells, text ids

`validation/datasets/generated/ab_common_gap.csv`: the same DGP without 2003 for every
unit, `y` missing in 10 rows and `x` in 10 others (`n_rows_dropped` = 20), and a text id
`unit` (`u001`..`u100`) beside the numeric `id`. pgmm's 16 instrument columns include five
that are zero for every equation (they need 2003); p2a does not create them, so it reports
11 instruments and Sargan df 9. No entity has two equations two periods apart, so AR(2) is
undefined (`NA` in R, NaN with a warning in p2a).

| Quantity | `onestep` | `twostep` |
|---|---|---|
| coefficients | 0.16915295624053117, 0.24284723725337082 | 0.06980315496528208, 0.2329910419985047 |
| SE one-step Arellano-Bond (1991) | 0.1460820757030355, 0.06242153326884828 |  |
| SE one-step robust (vcovHC) | 0.13957570026897384, 0.05771761986118569 |  |
| sigma2 | 0.7351940170379627 |  |
| Sargan homoskedastic (p) | 16.23122715893541 (0.06220717687129794) |  |
| SE two-step uncorrected |  | 0.12182556435805915, 0.05418416557517542 |
| SE two-step Windmeijer |  | 0.14912630257294213, 0.0670697087245282 |
| Sargan/Hansen (df, p) | 14.528392616277394 (9, 0.10473170856047082) | 13.858201515361518 (9, 0.1274594339543556) |
| AR(1) z (p) | -1.5383375045112073 (0.12396611414923135) | -0.9731186864957517 (0.33049431776190535) |
| AR(2) z (p) | NA (NA) | NA (NA) |
| equations | 260 | 260 |
| instruments (non-zero; pgmm) | 11; 16 | 11; 16 |

### `dpd_ab_grunfeld`: Grunfeld, 173 instruments for 10 firms

`validation/datasets/grunfeld.csv`, `inv ~ lag(inv, 1) + value + capital | lag(inv,
2:99)`, index (`firm`, `year`). Both weighting matrices are singular: the one-step matrix
has numerical rank 133 (`rank_one_step_weight`) and the two-step matrix rank 9
(`rank_two_step_weight`) of 173, and pgmm uses `MASS::ginv`. The Sargan p-value of 1
reflects instrument proliferation with N = 10 firms, not a good fit.

| Quantity | `onestep` | `twostep` |
|---|---|---|
| coefficients | 0.6796680080285489, 0.10699466702845292, 0.11207891951856341 | 0.6719522974788301, 0.10853729180487748, 0.11679279002203646 |
| SE one-step Arellano-Bond (1991) | 0.04944109384201603, 0.007763415085420762, 0.018821724733061065 |  |
| SE one-step robust (vcovHC) | 0.10402577648126722, 0.013773015788091173, 0.0381992198503537 |  |
| sigma2 | 1009.9618092108349 |  |
| Sargan homoskedastic (p) | 256.5467086366109 (1.9966517839977355e-05) |  |
| SE two-step uncorrected |  | 0.03383812334198877, 0.0020700879953147004, 0.017610892245971028 |
| SE two-step Windmeijer |  | 0.1220817501058947, 0.014836284415981081, 0.04971262563204494 |
| Sargan/Hansen (df, p) | 8.99907721785226 (170, 1) | 8.198968776259962 (170, 1) |
| AR(1) z (p) | -1.3410290358385935 (0.17991102558357105) | -1.3642837339379748 (0.17247828329506745) |
| AR(2) z (p) | -1.362302243133932 (0.17310252002619453) | -0.6931457052533437 (0.48821811732074377) |
| equations | 180 | 180 |
| instruments (non-zero; pgmm) | 173; 173 | 173; 173 |

System GMM (`transformation = "ld"`) on the same model: pgmm gives lag(inv, 1)
0.904434323720773, value 0.0210772058500476, capital 0.02408375709498078 (two-step, robust
SEs 0.08635876354389636, 0.01047086634052574, 0.03152250547905452; Sargan 6.519993993604108
(190), p 1; AR(1) -1.378238393462307; AR(2) -1.413475217229229), with both weighting
matrices rank-deficient (ranks 137 and 9 of 193 instruments). p2a computes it: the
estimator written out in R during planning reproduced pgmm to 8.3e-13 (one-step) and
6.5e-13 (two-step) in the coefficients and 3.3e-10 in the SEs, and p2a's two-step fit gives
0.9044343237199635, 0.021077205850118123, 0.024083757095145986 (Windmeijer SEs
0.08635876356770433, 0.010470866342362207, 0.03152250548453445; ranks 137 and 9). No case
pins it.

## Tests

Registered tests (`panel_gmm_difference`), every compared number against the case through
`p2a_validation::Check`:

- `econometrics::panel::utils::tests::test_validate_difference_equations_gaps_layout`: the
  equation and instrument counts of `dpd_ab_gaps` (553 equations, 22 instruments, 442 with
  two lags, 7 collapsed) and the layout of id 9's equations read from the CSV;
- `econometrics::panel::dynamic_panel::tests::test_validate_gmm_ab_balanced`,
  `test_validate_gmm_ab_balanced_variants`, `test_validate_gmm_ab_gaps`,
  `test_validate_gmm_ab_common_gap` (with numeric and text ids, the dropped rows against
  `n_rows_dropped`, the missing-value note as the last text line) and
  `test_validate_gmm_ab_grunfeld` (with the weighting-matrix ranks): every quantity above
  for the four fits (one-step classical and robust, two-step uncorrected and Windmeijer)
  and the variants;
- `tier2::wp19_dynamic_panel::test_validate_mcp_panel_gmm_ab_balanced` (p2a-mcp): the
  `panel_gmm` text at print precision, two-step and one-step classical;
- `tier2::wp19_dynamic_panel::test_validate_cli_panel_gmm_ab_balanced` (p2a-cli, `-F
  json` at full tolerance, with `--step`, `--vcov`, `--lags` and `--collapse`) and
  `tier2::wp19_dynamic_panel::test_validate_cli_panel_gmm_text` (the text at 6 decimals).

Property tests, not registered as R-pinned (no value is compared with R; they load
`validation/datasets/generated/ab_balanced.csv` or `ab_common_gap.csv` without a case):

- `test_difference_equations_duplicate_is_an_error`, `test_gmm_duplicate_entity_time_is_an_error`,
  `tier2::wp19_dynamic_panel::test_mcp_panel_gmm_duplicate_row_is_a_tool_error`,
  `tier2::wp19_dynamic_panel::test_cli_panel_gmm_duplicate_row_error`: a repeated
  (entity, time) row is refused ("Entity '1' has 2 rows for time 2002");
- `test_difference_equations_need_steps_of_one`, `test_gmm_lags_use_time_steps_of_one`:
  biennial years form no equation, and the error says to recode them;
- `test_gmm_fractional_time_is_an_error`: a fractional time is refused;
- `test_gmm_scale_equivariance`: x in millions gives the same lag coefficient and the x
  coefficient and SE scaled by 1e-6 (to 1e-9), with no rank warning;
- `tier3::wp58_models::test_validate_mcp_panel_gmm_system`: `panel_gmm` with
  `transform = "system"` (and `"ld"`) pins the two-step and one-step system-GMM rows of
  `dpd_sys_balanced` through MCP, prints the equation counts without the old ECON-30
  warning, and refuses unknown transforms and steps;
- `utils::tests::test_gmm_equations_date_time_hint`: yearly dates (365 or 366 days apart)
  form no equation, and the error says to recode them as consecutive period numbers;
- `test_gmm_onestep_rank_warning_names_the_sargan_weight`: on Grunfeld, one-step fits name
  the rank-deficient two-step weight as the Sargan/Hansen weighting matrix, without the
  two-step advice, which two-step fits keep;
- `test_gmm_transform_parses_aliases_and_rejects_unknown`,
  `test_gmm_step_parses_aliases_and_rejects_unknown`: the transform and step spellings, and
  the error for an unknown one;
- `test_na_gmm_int64_equals_float64`, `test_na_gmm_text_outcome_is_non_numeric`,
  `test_na_gmm_absent_column`, `test_na_gmm_missing_ids_and_times_are_dropped`: integer
  and float columns agree bit for bit, a text y is `NonNumericColumn` and an absent column
  `ColumnNotFound`, and rows with a missing id or time are dropped like any other;
- Tier 0's `test_gmm_rejects_max_lag_below_min_lag`, `test_gmm_huge_lags_is_an_error_not_a_panic`,
  `test_gmm_instrument_matrix_respects_alloc_ceiling`, `test_gmm_stops_when_cancelled` and
  `test_gmm_scaling_many_entities` (release only), and
  `econometrics::panel::tests::test_arellano_bond_convenience`.

## System GMM (Blundell-Bond 1998)

`run_gmm` with `GmmConfig.transform = GmmTransform::System` computes system GMM as
`plm::pgmm(transformation = "ld")` with its defaults `fsm = "full"` and `effect =
"individual"`.

### Estimator and pgmm conventions

- **Equations.** Each entity's difference equations (as for difference GMM) are stacked
  with its level equations, difference rows first: the level equation of (entity, t) is y_t
  on y_t-1, ..., y_t-lags and x_t, and exists where y is observed at t, ..., t-lags and t
  is at least max(lags + 1, min_lag) - 1 periods after the first period. `n_obs` counts
  both kinds (`n_obs_diff`, `n_obs_level`), as pgmm's "Number of Observations Used".
- **Instruments.** The level equation at t is instrumented by Delta y_t-1 = y_t-1 - y_t-2
  where both are observed, one column per period (one column with `collapse`), whatever
  `min_lag`. x is its own instrument in differences for the difference rows and in levels
  for the level rows (pgmm's `bdiag(Z1, Z2)`), so L = difference GMM + level GMM + 2 k_x
  columns.
- **No intercept or time dummies** (`effect = "individual"`). The level equations have no
  intercept (plm::pgmm `effect = "individual"`); with x in levels as its own instrument they
  are valid only when E[eta_i] = 0 or x has mean zero; Stata's xtabond2 adds `_cons`.
- **Weights.** One-step: (sum_i Z_i' H_i Z_i)^-1 with pgmm's `FSM(., "full")` H: the
  tridiagonal difference block (2, -1), the identity level block, and the cross term +1
  between the difference and the level equation of the same period, -1 between the
  difference equation at t and the level equation at t - 1. Two-step: (sum_i Z_i' e_i e_i'
  Z_i)^-1 from the one-step residuals, as for difference GMM, with the same generalized
  inverse rule.
- **Standard errors.** One-step robust (`vcovHC.pgmm`), two-step uncorrected or Windmeijer
  (2005). sigma2 = sum e^2 / (2 (n_obs_diff - k)) from the one-step residuals of the
  difference equations; it is reported, but not used for standard errors (see below).
- **Tests.** Sargan/Hansen as `plm::sargan`, df = L - k. The AR(1)/AR(2) m-tests as
  `plm::mtest`, on the differenced residuals only (pgmm keeps the first `Kt - 1` rows of
  each entity and zeroes the level rows).

### R code

`validation/r/cases/econ_sysgmm.R` ([source](../r/cases/econ_sysgmm.R)), cases
`dpd_sys_balanced`, `dpd_sys_gaps` and `dpd_sys_empluk`. Each fits `pgmm(...,
effect = "individual", transformation = "ld")` one-step and two-step and reads the SEs from
`summary(robust = FALSE)` and `summary(robust = TRUE)`, the equation counts from pgmm's
residuals, and sigma2 from the one-step difference residuals. The non-robust one-step SEs
(`onestep_se`, pgmm's unscaled `m$B1`) and the Wald statistics are recorded and not pinned.
The `lags3_spec_*` and `minlag4_spec_*` keys come from `.sys_spec`, the estimator written
out in R, which the case first checks against pgmm on the default, two-lag and `min_lag =
3` fits. `dpd_sys_empluk` writes `validation/datasets/generated/sysgmm_empluk.csv`
(`plm::EmplUK` with `lemp`, `lwage`, `lcap` the logs of employment, wage and capital; 1031
rows, 140 firms, 1976-1984).

### Values

`dpd_sys_balanced` (`ab_balanced.csv`, `y ~ lag(y, 1) + x | lag(y, 2:99)`):

| Quantity | `onestep` | `twostep` |
|---|---|---|
| coefficients | 0.822771528612361, 0.30670618306258884 | 0.8194537186389175, 0.2997533922594534 |
| SE one-step robust (vcovHC) | 0.027320976059039045, 0.041764270438180796 |  |
| sigma2 | 1.4401318877737306 |  |
| SE two-step uncorrected |  | 0.021471081529256232, 0.03636035462874532 |
| SE two-step Windmeijer |  | 0.035723644003136414, 0.0539568926255106 |
| Sargan/Hansen (df, p) | 40.146725761163744 (27, 0.04963971040629556) | 40.08030035403464 (27, 0.05035732750818256) |
| AR(1) z (p) | -5.881080194063606 (4.075976103952484e-09) | -6.409008658808723 (1.4646887678573905e-10) |
| AR(2) z (p) | 0.004695445769111646 (0.9962535900810587) | -0.001233961287909219 (0.9990154415896069) |
| equations (differenced + level) | 1300 (600 + 700) | 1300 (600 + 700) |
| entities; instruments | 100; 29 | 100; 29 |

| Variant (pgmm-pinned) | coefficients | SE (Windmeijer or one-step robust) | Sargan (df) | equations; instruments |
|---|---|---|---|---|
| `lags2_twostep`, `lag(y, 1:2)` | 0.7368965251942556, 0.13090302697822137, 0.3113979598258415 | 0.04292180051013481, 0.044844528203951385, 0.052062499282716924 | 30.51741283159093 (25) | 1100 (500 + 600); 28 |
| `maxlag3_twostep`, `lag(y, 2:3)` | 0.7754697608200896, 0.3099686672475889 | 0.03853162869957186, 0.05202876375183004 | 27.262566188077034 (17) | 1300; 19 |
| `minlag3_onestep`, `lag(y, 3:99)` | 0.7471714147040462, 0.3119214733144682 | 0.04614527133814925, 0.04256093288003956 | 30.597286885044866 (21) | 1100 (500 + 600); 23 |
| `collapse_twostep` | 0.7209148422479621, 0.3038418175872592 | 0.05064007304659357, 0.042879263596668146 | 6.564539943650787 (7) | 1300; 9 |
| `nox_onestep`, no x | 0.8222549260440193 | 0.027743932321347996 | 40.24311497797024 (26) | 1300; 27 |

`minlag3_onestep` sigma2 1.355930797057432, `nox_onestep` sigma2 1.5490978881916444.

| Variant (pinned to `.sys_spec`) | one-step coefficients | two-step coefficients | two-step Windmeijer SE | two-step Sargan | equations; instruments |
|---|---|---|---|---|---|
| `lags3_spec`, 3 lags | 0.6823946139175225, 0.07849583437903809, 0.09724696259278708, 0.3183205081696201 | 0.6869588743703935, 0.08593549657995833, 0.09977245086217795, 0.3335396065256411 | 0.06389356058319648, 0.05899080007754918, 0.05516705118816765, 0.05650419917498504 | 24.896962804697843 | 900 (400 + 500); 25 |
| `minlag4_spec`, min_lag 4 | 0.6991752653753088, 0.3236478951628366 | 0.6866577847126246, 0.3218254229448384 | 0.09953987440185884, 0.06175884623471312 | 25.400291262580367 | 900 (400 + 500); 17 |

`dpd_sys_gaps` (`ab_gaps.csv`: ids divisible by 9 miss 2004, ids divisible by 13 start in
2002; id 9 has level equations at 2001, 2002, 2003, 2006 and 2007, of which 2002, 2003 and
2007 have a level instrument):

| Quantity | `onestep` | `twostep` | `lags2_twostep` | `collapse_twostep` |
|---|---|---|---|---|
| coefficients | 0.8083649768862912, 0.3222860051423028 | 0.8252338934470143, 0.3120869394422262 | 0.7434769384411624, 0.1710358674000927, 0.3120629857776108 | 0.6969852538407156, 0.3257326446129133 |
| SE one-step robust (vcovHC) | 0.030483014938680154, 0.043068173842383074 |  |  |  |
| sigma2 | 1.4400761395143025 |  |  |  |
| SE two-step uncorrected |  | 0.025227107725092856, 0.0369811961916684 | 0.03159108210545694, 0.03998365555446948, 0.040958543242977395 | 0.05372702470725777, 0.041305426327512336 |
| SE two-step Windmeijer |  | 0.0406709345050405, 0.053698228810411766 | 0.046212695931643635, 0.0593922222428899, 0.05289003112109152 | 0.055997929924971115, 0.044317436938468006 |
| Sargan/Hansen (df, p) | 42.29152381483864 (27, 0.030800475883339395) | 41.79936019427313 (27, 0.03444511631784372) | 29.67505219601324 (25, 0.23668699201393834) | 4.837459193828679 (7, 0.6797910811348671) |
| AR(1) z | -5.578384358132303 | -6.053263312942498 | -5.011714571511001 | -5.365214687604275 |
| AR(2) z (p) | 0.2986312965592522 (0.765221383297685) | 0.302685146002729 (0.7621298188668887) | -0.5390038071642562 (0.5898842275828156) | 0.2857494239072926 (0.7750700474057157) |
| equations (differenced + level) | 1217 (553 + 664) | 1217 (553 + 664) | 995 (442 + 553) | 1217 (553 + 664) |
| instruments | 29 | 29 | 28 | 9 |

`dpd_sys_empluk` (`lemp ~ lag(lemp, 1) + lwage + lcap | lag(lemp, 2:99)`, index (`firm`,
`year`)):

| Quantity | `onestep` | `twostep` |
|---|---|---|
| coefficients | 0.7456410279393282, 0.10192324280488663, 0.20831291642498684 | 0.7379632375194216, 0.10317196014058094, 0.21597642070575596 |
| SE one-step robust (vcovHC) | 0.0627844634554638, 0.0288880089561445, 0.04784663251710622 |  |
| sigma2 | 0.010846797402085849 |  |
| SE two-step uncorrected |  | 0.024963497103525963, 0.012125288841971418, 0.01960128919094153 |
| SE two-step Windmeijer |  | 0.0646288566904284, 0.028639170446890516, 0.048257985107939334 |
| Sargan/Hansen (df, p) | 70.90449348613491 (36, 0.0004589438876224779) | 69.8754641054323 (36, 0.0006057443613799031) |
| AR(1) z (p) | -1.9581956560524971 (0.050207058666891055) | -2.0030574991247927 (0.04517111713583335) |
| AR(2) z (p) | -0.3671030450385006 (0.7135421560204294) | -0.348825415565596 (0.7272203816070718) |
| equations (differenced + level) | 1642 (751 + 891) | 1642 (751 + 891) |
| entities; instruments | 140; 39 | 140; 39 |

### Tests and tolerances

Registered tests (`panel_gmm_system` in `validation/methods/econ_sysgmm.json`), in
`econometrics::panel::dynamic_panel::tests` unless named otherwise:

- `test_validate_sysgmm_balanced_onestep`, `test_validate_sysgmm_balanced_twostep`,
  `test_validate_sysgmm_balanced_variants`, `test_validate_sysgmm_gaps`,
  `test_validate_sysgmm_empluk`: every quantity of the tables above (coefficients, SEs,
  sigma2, Sargan and its df and p, AR(1)/AR(2) and their p, the equation, entity and
  instrument counts); each fit also has full-rank weights, no warning, no homoskedastic
  Sargan, and two-step uncorrected and Windmeijer coefficients equal bit for bit;
- `test_validate_sysgmm_spec_lags3_minlag4`: the `lags3_spec_*` and `minlag4_spec_*` keys;
- `test_validate_sysgmm_differs_from_difference`: on `ab_balanced`, system and difference
  GMM differ (0.81945 against 0.37752), with their equation and instrument counts and
  output headers;
- `test_validate_sysgmm_onestep_classical_se_is_refused`: one-step classical SEs are an
  error for system GMM, two-step uncorrected SEs are not;
- `test_validate_sysgmm_scale_equivariance`: x in millions gives the same lag estimate and
  the x estimate and SE scaled by 1e-6 (to 1e-9), with full-rank weights;
- `test_validate_sysgmm_rejects_max_lag_below_min_lag`: `max_lag` below `min_lag` is an
  error on the system path (X-7's repro, `max_lag` 0 or 1);
- `econometrics::panel::utils::tests::test_validate_system_equations_balanced_layout`,
  `test_validate_system_equations_gaps_layout`: the stacked equations (counts, row kinds,
  times, the level rows' instruments read from the CSV digits).

Property tests, not registered: `utils::tests::test_system_h_weight_is_pgmm_fsm_full` (H
against `plm:::FSM(4, "full")`), and `test_sysgmm_scaling_many_entities` (release only:
50,000 entities, 5 periods, below 10 s).

Tolerances: `validation/README.md` with n = equations used. For n > 1000 (1100, 1217,
1300, 1642) coefficients `Tol::coef(n)` = 1e-10, SEs, sigma2 and statistics 1e-8; for n =
900 and 995, 1e-8 and 1e-6 relative; counts and df exact. The estimator written out in R
agrees with pgmm to 1.3e-12 relative on every pinned pgmm quantity.

### Differences from pgmm

- **`lags >= 3` or `min_lag >= 4`.** pgmm 2.6.7 builds its level-instrument matrix with
  `matrix(..., nrow = T - TL2)` from a vector of length (T - 2)^2, which is right only for
  TL2 <= 2; beyond it R recycles the vector (warning "data length [36] is not a
  sub-multiple or multiple of the number of rows [5]") and the estimates are not the
  system estimator. For lags 3 pgmm returns 0.72174827332, 0.06695246557, 0.09088908737,
  0.36034067886 against the estimator's 0.6869588743703935, 0.08593549657995833,
  0.09977245086217795, 0.3335396065256411. p2a applies pgmm's rule for TL2 <= 2 (the level
  instrument at t is y_t-1 - y_t-2) to every TL2, and these values are pinned to `.sys_spec`,
  not to plm.
- **One-step classical SEs: refused.** pgmm's `summary(robust = FALSE)` one-step SEs are
  sqrt(diag(m$B1)), without sigma2 and with a 1/N factor, and H is the error covariance of
  the stacked errors only when var(eta_i) = 0. On EmplUK the homoskedastic Sargan statistic
  this H implies would be 291.2 against the robust 70.9. `step = OneStep, robust = false`
  is an `InvalidSpecification` for system GMM, and `sargan_homoskedastic` is `None`.
- **Instrument columns.** As for difference GMM, all-zero instrument columns are not
  created. None occur in these cases.
- **Missing values.** Rows with a missing value are dropped (`na.omit`), as for difference
  GMM.

### Conventions others use

Stata's xtabond2 (Roodman 2009) includes `_cons` in the level equations and dates the level
instrument at `t - min_lag + 1`. pgmm and p2a use t - 1 and no constant, so results differ
from xtabond2 when `min_lag > 2` or E[eta_i] != 0.

### Not covered (system GMM)

Time dummies (`effect = "twoways"`), an intercept in the level equations, `fsm = "GI"`, and
predetermined or endogenous x.

## Numerical Precision Summary

`validation/README.md`, n = 180-600 equations: coefficients `Tol::coef(n)` (1e-8
relative), SEs, sigma2 and statistics `Tol::se(n)`/`Tol::stat(n)` (1e-6), p-values
`Tol::p_value(n)`; counts and df exact. Grunfeld (n = 180) uses these tolerances with the
relative part of the SE, sigma2, statistic and p-value bounds tightened to 1e-7: with its
rank-deficient weighting matrices (eigen- vs SVD-based truncation), the
coefficients agree to 3e-12, the SEs to 1.6e-10 and the statistics and p-values to 6e-11.
The implementation agrees with pgmm to about 1e-10 relative on every pinned quantity (2e-10 on
Grunfeld). The MCP text is compared at `Tol::printed(4)`, the CLI text at
`Tol::printed(6)`.

## Differences from pgmm

1. Instrument columns that no equation observes are not created, so the instrument count
   and the Sargan df differ from pgmm's when a period is unobserved (pinned to the
   non-zero count; `dpd_ab_common_gap`).
2. With `max_lag`/`min_lag` and a period that no unit observes, pgmm counts instrument lags
   in positions of its time grid, p2a in time units.
3. p2a drops rows with a missing value in y, x, entity or time (`na.omit`), while pgmm
   (`na.pass`) still uses an observed y_t of a row whose x_t is missing as an instrument;
   the cases compute on the complete rows.
4. pgmm decides singularity by an absolute eigenvalue threshold (1e-9), p2a on the
   equilibrated matrix (the same decision on every case).
5. p2a reports Arellano-Bond (1991) classical one-step SEs, where pgmm's non-robust
   one-step `vcov()` is unscaled.
6. The m-tests always use the robust covariance (pgmm's `summary` default).

## Not covered

- Time dummies (`effect = "twoways"`).
- Predetermined or endogenous x (GMM-style instruments for x).

## Performance Comparison

No timing is claimed here. Speed comparisons with R come from the benchmark pipeline in the
paper repository (`performance/` in `../prompt2analytics-paper`), which runs the Rust
benchmarks of `crates/p2a-core/benches/` against R.

## References

1. Arellano, M., & Bond, S. (1991). "Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations." *Review of Economic Studies*, 58(2), 277-297.

2. Blundell, R., & Bond, S. (1998). "Initial conditions and moment restrictions in dynamic panel data models." *Journal of Econometrics*, 87(1), 115-143.

3. Windmeijer, F. (2005). "A finite sample correction for the variance of linear efficient two-step GMM estimators." *Journal of Econometrics*, 126(1), 25-51.

4. R package `plm` (Croissant & Millo): https://cran.r-project.org/package=plm
