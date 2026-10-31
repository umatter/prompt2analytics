# Validation: Arellano-Bond difference GMM

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
`transform = System` (Blundell-Bond) is not implemented yet (ECON-30): it runs difference
GMM, reports `Difference` and warns.

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

System GMM (`transformation = "ld"`, two-step, robust) on the same model, kept as an R-only
reference for ECON-30 (not compared; no p2a value is claimed): lag(inv, 1)
0.904434323720773, value 0.0210772058500476, capital 0.02408375709498078; robust SEs
0.08635876354389636, 0.01047086634052574, 0.03152250547905452; Sargan 6.519993993604108
(190), p 1; AR(1) -1.378238393462307 (p 0.1681296969832509); AR(2) -1.413475217229229 (p
0.1575160432179447).

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
- `test_gmm_system_transform_reports_difference`,
  `tier2::wp19_dynamic_panel::test_mcp_panel_gmm_system_warns`: `transform = System` gives
  the difference estimates, labelled as such, with the ECON-30 warning;
- `test_na_gmm_int64_equals_float64`, `test_na_gmm_text_outcome_is_non_numeric`,
  `test_na_gmm_absent_column`, `test_na_gmm_missing_ids_and_times_are_dropped`: integer
  and float columns agree bit for bit, a text y is `NonNumericColumn` and an absent column
  `ColumnNotFound`, and rows with a missing id or time are dropped like any other;
- Tier 0's `test_gmm_rejects_max_lag_below_min_lag`, `test_gmm_huge_lags_is_an_error_not_a_panic`,
  `test_gmm_instrument_matrix_respects_alloc_ceiling`, `test_gmm_stops_when_cancelled` and
  `test_gmm_scaling_many_entities` (release only), and
  `econometrics::panel::tests::test_arellano_bond_convenience`.

## Numerical Precision Summary

`validation/README.md`, n = 180-600 equations: coefficients `Tol::coef(n)` (1e-8
relative), SEs, sigma2 and statistics `Tol::se(n)`/`Tol::stat(n)` (1e-6), p-values
`Tol::p_value(n)`; counts and df exact, Grunfeld (n = 180) included: with its rank-deficient
weighting matrices (eigen- vs SVD-based truncation), the coefficients agree to 3e-12 and the
SEs to 1.6e-10. The
implementation agrees with pgmm to about 1e-10 relative on every pinned quantity (2e-10 on
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

- System GMM (Blundell-Bond, `transformation = "ld"`; ECON-30, Tier 3).
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
