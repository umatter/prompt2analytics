# Validation: Panel unit-root tests (data extraction, lag order and statistics)

## Method Overview

`run_panel_unit_root` (`crates/p2a-core/src/econometrics/panel_unit_root.rs`) runs the
Levin-Lin-Chu, Im-Pesaran-Shin, Fisher (Maddala-Wu) and Hadri tests on a long-format
panel. Each test is built on per-unit regressions over each unit's series; this document
covers how those series are read from the data and validates the per-unit ADF regression
that the IPS and Fisher tests average or combine.

**Status**: R-pinned: the data extraction, the per-unit ADF t-statistic (`urca::ur.df`,
`plm::purtest`), the per-unit lag orders of AIC, SIC and Hall selection, and the LLC, IPS,
Maddala-Wu and Hadri statistics and p-values (`plm::purtest`), the per-unit MacKinnon (1994)
p-values (`plm:::padf(p.approx = "MacKinnon1994")`) and Hadri's long-run option
(`tseries::kpss.test`). Known deviations: ECON-44 (MacKinnon 1994 instead of plm's default
1996 tables; the surface held at its turning points) and ECON-88 (defaults).

## Data requirements and conventions

- **Units** are read as a factor: numeric or text ids. Units are ordered as R's `factor()`
  orders them (numbers numerically, text in byte, C-locale, order); `unit_statistics` and
  `unit_p_values` follow that order. For numeric ids this is the order the extraction used
  before, so results on numeric ids with integer times are unchanged.
- **Time** must hold whole numbers or dates (integers of any width, whole-number floats,
  `Date`); a fractional value is `InvalidSpecification` ("must hold whole numbers").
  Before, times were cast to integers and truncated, so quarter values such as 2000.25
  collided and overwrote each other. A `Date` column counts days, so dates more than a day
  apart (monthly, quarterly or yearly dates) are reported as gaps; recode them as period
  numbers first.
- **Repeated (unit, time)** rows are an error: "Unit '1' has 2 rows for time 2002; panel
  unit-root tests need one row per (unit, time)." Before, the later row silently replaced
  the earlier one.
- **Gaps.** A unit's observations are used in time order as if consecutive, as
  `plm::purtest` does (its IPS statistic is the same on a gapped panel and on the panel
  relabelled with consecutive periods). The result warns: "{n} of {N} unit(s) have gaps in
  time (first: unit '{label}' between {t1} and {t2}); their observations are joined in
  time order as if consecutive, as plm::purtest does."
- **Short units.** Units with fewer than 3 observations are dropped with a warning naming
  them (at most five labels): "Dropped 1 unit(s) with fewer than 3 observations: '7'".
  Before, they vanished silently. No unit left is `InsufficientData`.
- **Missing values.** Rows with a missing value (null or NaN) in the variable, the unit or
  the time are dropped (`na.omit`) and reported in `missing`, in `warnings` and as the last
  line of the text ("Missing values: 10 of 700 rows dropped because of missing values (y:
  10)"). An infinite value is `NonFiniteValues`; a text variable is `NonNumericColumn`.
  Before, text unit ids became null in a non-strict cast, every row was dropped and the
  call failed with "Empty dataset".

## R reference

Case `panel_pur_extraction` of `validation/r/cases/econ_dynamic_panel.R`
([source](../r/cases/econ_dynamic_panel.R)), R 4.3.3, urca 1.3.4, on
`validation/datasets/generated/ab_common_gap.csv` (written by case `dpd_ab_common_gap`):
100 units with text ids `u001`..`u100` (and the same units as numeric `id`), years
2000-2007 without 2003, which no unit observes, and 10 missing `y` cells. Every unit has a
gap, 690 rows are complete in (`unit`, `year`, `y`).

```r
dd <- d[complete.cases(d[, c("unit", "year", "y")]), ]
units <- split(dd, factor(dd$unit, levels = sort(unique(dd$unit), method = "radix")))
tau <- vapply(units, function(u) {
  u <- u[order(u$year), ]
  unname(urca::ur.df(u$y, type = "drift", lags = 1)@teststat[1])
}, 0)
```

| Quantity | R |
|---|---|
| `unit_tau` (first three, u001..u003) | -1.9259975618874472, -1.633627802064327, -1.8328340217447496 |
| `unit_tau` (last, u100) | -1.1647290630151956 |
| `n_panels` | 100 |
| `avg_t` | 6.9 |
| `n_rows_dropped` | 10 |

## Tests

Registered tests (registry entry `panel_unit_root_extraction` in
`validation/methods/econ_dynamic_panel.json`):

- `econometrics::panel_unit_root::tests::test_validate_panel_pur_extraction_common_gap`:
  `run_panel_unit_root(.., "y", "unit", "year", IPS, intercept, lags 1)`: the 100 per-unit
  ADF taus against `unit_tau` (`Tol::stat(690)`), `n_panels`, `avg_t` and
  `n_rows_dropped` exactly; the gap warning, the missing-value note as the last text line,
  and the numeric ids `id` giving the same taus bit for bit.
- `tier2::wp19_dynamic_panel::test_validate_mcp_panel_unit_root_text_ids` (p2a-mcp): the
  `panel_unit_root` text on the text ids, `n_panels`, `avg_t` (1 decimal) and
  `n_rows_dropped` from the printed lines, and the gap warning;
- `tier2::wp19_dynamic_panel::test_validate_cli_panel_unit_root_text_ids` (p2a-cli): `p2a
  panel unit-root -F json` (`unit_statistics` against `unit_tau`, `n_panels`,
  `avg_time_periods`, `missing.n_dropped`, the gap warning) and the text's `Warnings:`
  block.

Property tests, not registered as R-pinned (no value is compared with R; they load
`validation/datasets/generated/ab_balanced.csv` or `ab_common_gap.csv` without a case):

- `test_pur_fractional_time_is_an_error`: quarter times (2000.25, ...) are refused;
- `test_pur_duplicate_unit_time_is_an_error`: a repeated (unit, time) row is refused;
- `test_pur_short_units_dropped_with_warning`: a unit with 2 observations is dropped and
  named;
- `test_na_pur_int64_and_float64_time`: Int64 and Float64 years give bit-identical
  statistics;
- `test_na_pur_text_variable_is_non_numeric`: a text variable is `NonNumericColumn`.

## Lag order

The ADF-type tests (LLC, IPS, Fisher) regress each unit's first difference on its lagged
level, the deterministic terms (none, an intercept, an intercept and a trend) and `p`
lagged differences, on the unit's rows `p + 2..T`.

- **Default** (neither `lags` nor `lag_selection`): `floor(T_max^(1/3))` lags for every unit,
  `T_max` the longest unit, capped by `max_lags`. This is unchanged from before Tier 3 (it was
  described as automatic selection; ECON-88). The output labels it: `Lags:       3 (fixed:
  floor(T_max^(1/3)))`, or `(fixed: min(floor(T_max^(1/3)), max_lags))` when the cap binds.
- **Fixed** (`lags = Some(p)` or `lag_selection = fixed`): `p` lags for every unit,
  labelled `(fixed by the user)`.
- **Per unit** (`lag_selection = aic | sic | hall`), as `plm::purtest(lags = "AIC" | "SIC" |
  "Hall", pmax, fixedT = TRUE)`: every order `0..=pmax` is fitted on the common sample
  `t = pmax + 2..T`. AIC and SIC pick the smallest `ln(rss/n) + c K / n` (`c = 2` or
  `ln n`, `K` the number of columns; a tie goes to the larger order, as plm's `which.min`
  over the descending list). Hall's rule picks the largest order whose last lagged
  difference has `|t| >= 1.96`, else 0. `pmax` defaults to `max_lags`, else 10 (plm's
  default). The label reads `AIC per unit, pmax = 4`, and a line `per unit: min 0, max 4,
  mean 2.33` follows when the orders differ. `unit_lags` holds each unit's order,
  `lags_used` the largest.
- **`dfcor`** chooses the residual variance of the ADF regression: `rss/(n - K)` (true, the
  default here and `urca::ur.df`'s convention) or `rss/n` (false, plm's default). Hall's
  rule and the per-unit t depend on it, AIC and SIC do not; the output prints it.
- Spellings: `lag_selection` fixed | aic | sic (bic, schwarz) | hall (t_sig, tsig); the test
  llc (levinlin, levin_lin_chu) | ips (im_pesaran_shin) | madwu (maddala_wu, fisher) |
  hadri; the model none (nc) | intercept (drift, c, const) | trend (ct). Case and the
  separators ' ', '-', '.', '_' do not matter; anything else is an error listing the names.
- Contradicting options are errors: `lags` with a criterion ("use pmax"), `lags` and
  `fixed (p)` that differ, `pmax = 0` with a criterion, `pmax` and `max_lags` that differ,
  `pmax` without a criterion, and `lag_selection` or `pmax` for Hadri (it has no lagged
  differences).

**Short and degenerate units.** A unit needs `T >= 3 + n_det + 2p` observations for `p`
lags (`n_det` = 0, 1 or 2), so that its ADF regression keeps a residual degree of freedom;
lag selection needs this for `pmax`. A shorter unit, or one whose regressors are collinear
(an exactly linear series), is skipped and named in one warning: "Skipped 1 unit(s) whose
ADF regression cannot be estimated: '10' (6 observations; 2 lag(s) with an intercept need at
least 8)" or "... '7' (rank deficient: rank 2 of 3)". `n_panels`, `unit_ids` and the
statistics then count the units used. With no unit left the test returns the first unit's
error ("Im-Pesaran-Shin: no unit can be used: ADF regression with 2 lag(s) and an intercept
needs at least 8 observations"). Before Tier 3 a design with no residual degree of freedom
underflowed (a panic in a debug build, |t| of order 1e9 in a release build) or gave
t = -0.0 at an exact fit, the LLC auxiliary regressions fell back to the raw series, and
rank-deficient units were pseudo-inverted (ECON-42). The Fisher p-value is the exact
chi-square upper tail (it was the Wilson-Hilferty approximation) and Hadri's the normal
upper tail.

### R reference: `purtest_lag_selection`

`validation/r/cases/econ_purtest.R`, R 4.3.3, plm 2.6.7, on
`validation/datasets/generated/purtest_panel.csv` (12 units `id` 1..12, `t` 1..40, `y`):

```r
p <- pdata.frame(d, index = c("id", "t"))
purtest(p$y, test = "ips", exo = "intercept", lags = "AIC", pmax = 4, dfcor = FALSE, fixedT = TRUE)
# test = "ips" | "levinlin" | "madwu"; lags = "AIC" | "SIC" | "Hall"; dfcor = FALSE | TRUE;
# plus exo = "trend" (AIC, dfcor TRUE) and exo = "none" (SIC, dfcor TRUE)
```

| Rule (intercept, pmax 4) | Per-unit lags, units 1..12 |
|---|---|
| AIC (dfcor FALSE and TRUE) | 4, 1, 2, 3, 1, 2, 3, 4, 1, 0, 2, 1 |
| SIC (dfcor FALSE and TRUE) | 2, 1, 1, 1, 1, 2, 2, 0, 1, 0, 1, 1 |
| Hall, dfcor FALSE | 4, 1, 1, 1, 1, 2, 2, 4, 1, 0, 1, 1 |
| Hall, dfcor TRUE | 2, 1, 1, 1, 1, 2, 2, 4, 1, 0, 1, 1 |
| AIC, trend, dfcor TRUE | 4, 1, 2, 3, 1, 2, 1, 3, 3, 1, 1, 1 |
| SIC, none, dfcor TRUE | 2, 1, 2, 1, 1, 2, 2, 0, 0, 0, 1, 0 |

The lags are the same for IPS, LLC and Maddala-Wu. Per-unit ADF t (`trho`), first three
units: AIC -2.376110959, -3.643804459, -2.082619119; SIC -1.362860564, -3.643804459,
-3.584969124; AIC with dfcor TRUE -2.162876362, -3.497013322, -1.966826083 (the case file
holds all twelve for each rule, with 16 digits). An R transcription of the selection
reproduces every lag vector; its smallest decision margin is 2.5e-3 in the criterion and
4.8e-3 in |t| - 1.96, so rounding cannot flip a choice.

Tests (method `panel_unit_root_lag_selection` in `validation/methods/econ_panel_t3.json`):
- `econometrics::panel_unit_root::tests::test_validate_purtest_lag_selection_ips`: the lags
  (exact) and per-unit t (`Tol::stat(480)`) for the six rule and dfcor combinations, the
  trend variant, `n_panels`, `lags_used`, `unit_ids`, the label and `dfcor`;
- `econometrics::panel_unit_root::tests::test_validate_purtest_lag_selection_llc`: the LLC
  lags for the six combinations;
- `econometrics::panel_unit_root::tests::test_validate_purtest_lag_selection_madwu`: the
  Maddala-Wu lags and per-unit t for the six combinations and the no-deterministics
  variant.

Property tests (no case loaded): `test_lag_selection_rules_give_different_lags`,
`test_default_lag_rule_is_unchanged` (the default gives 3 lags on this panel, bit for bit the
fit with `fixed (3)`), `test_lag_selection_config_conflicts_are_errors`,
`test_panel_unit_root_test_parses_aliases_and_rejects_unknown`,
`test_panel_model_parses_aliases_and_rejects_unknown`,
`test_lag_selection_parses_aliases_and_rejects_unknown`, `test_adf_rejects_short_panels`,
`test_adf_and_llc_skip_rank_deficient_units`.

Deviations (registry): the Maddala-Wu statistic, p and per-unit p (ECON-44; plm's default
`p.approx = "MacKinnon1996"`, which p2a does not use, see "Statistics"; the six statistics are
asserted within 2% of plm's default ones), and the defaults (ECON-88): p2a's default is a fixed
`floor(T_max^(1/3))` lags for every unit with `dfcor = true`, while `plm::purtest` defaults to
`lags = "SIC"`, `pmax = 10` and `dfcor = FALSE`. Default-config outputs are not pinned to plm's
defaults; the pins use explicit `lags = "AIC" | "SIC" | "Hall"`, `pmax = 4` and both `dfcor`
values. The IPS and LLC statistics and p-values of the six rule and dfcor combinations are
pinned (`Tol::stat(480)`, `Tol::p_value(480)`); for example LLC with AIC
-0.4160604664203227 (p 0.3386828788009173), IPS with AIC -1.3507583721703489 (p
0.08838642343843883).

## Statistics

Notation: unit series of length `T_i`, lag order `p_i` (fixed or selected), `nd` = 0, 1 or 2
deterministic terms, ADF rows `t = p_i + 2..T_i`, `n_i = T_i - p_i - 1` rows, `K_i = 1 + nd +
p_i` columns. The ADF residual variance is `rss/(n - K)` with `dfcor` and `rss/n` without.

- **Levin-Lin-Chu** (`purtest(test = "levinlin")`). Balanced panels only (checked after
  skipping units that cannot be estimated): "the Levin-Lin-Chu test needs a balanced panel
  ...; the Im-Pesaran-Shin and Maddala-Wu tests allow unbalanced panels". Per unit, the
  auxiliary residuals `e_i` (Δy) and `v_i` (y_{t-1}) on the deterministic terms and lagged
  differences give the ADF residual sd `sigma_i` (rss `e'e - (v'e)^2 / v'v`, Frisch-Waugh-Lovell);
  a lagged level the other regressors explain (`|v| < 1e-7 |y_{t-1}|`) skips the unit as rank
  deficient. The long-run sd of Δy (demeaned, detrended on (1, t) or raw by model) uses the
  Bartlett kernel with `q = round(3.21 T^(1/3))` lags (R's half-to-even rounding), divided by
  T - 1; for T <= 7, q exceeds T - 2 and is capped there with a warning (plm indexes out of
  range there, so there is no R value). `S_N` = mean of long-run sd / `sigma_i`. The
  normalised residuals `e_i / sigma_i` and `v_i / sigma_i` are pooled into one regression
  without intercept: `rho`, its sd (`rss/(N_obs - 1)` with dfcor) and `t_rho`. With `T~ = T -
  mean(p_i) - 1` and `sigma~^2 = rss / (N T~)`, `z = (t_rho - N T~ S_N / sigma~^2 sd(rho)
  mu*) / sigma*`, `p = Phi(z)`. `(mu*, sigma*)` are LLC (2002) Table 2 as plm stores them
  (`plm:::adj.levinlin`), interpolated linearly in T; below T = 25 or above 500 the end row
  is used with a warning. The statistic is scale invariant.
- **Im-Pesaran-Shin** (`purtest(test = "ips")`, W-tbar). Each unit's E and Var of t come from
  IPS (2003) Table 3 as plm stores it (`plm:::adj.ips.wtbar`) at `L_i = T_i - p_i - 1` and lag
  order `min(p_i, 8)`, interpolated linearly in L; units with L below 10 or above 100 use the end
  row, and units with more than 8 lags the 8-lag row, each with a warning. An untabulated
  (lags, L) (5 lags below L = 20, 6 to 8 lags below 25), where plm returns NA, is an error.
  `W = sqrt(N) (mean t_i - mean E_i) / sqrt(mean V_i)`, `p = Phi(W)`. Model `none` is refused
  (the table has no row for it; plm refuses it). Unbalanced panels are allowed.
- **Maddala-Wu** (`purtest(test = "madwu", p.approx = "MacKinnon1994")`). `-2 sum ln p_i`
  against chi-square(2N), upper tail. Each `p_i` is MacKinnon's (1994) asymptotic response
  surface for the Dickey-Fuller t, as `plm:::padf`: `Phi(b0 + b1 t + b2 t^2)` up to t = -1.04,
  -1.61, -2.89 (none, intercept, trend), `Phi(c0 + c1 t + c2 t^2 + c3 t^3)` above, with plm's
  (rescaled) coefficients. The surface takes no degrees of freedom (it is asymptotic), so the
  ADF regression's df does not enter. Outside the range where the polynomials rise (below the
  quadratic's vertex, t = -19.05, -18.83, -16.18, and above the cubic's local maximum, t =
  2.74 with an intercept and 0.70 with a trend) t is held at the turning point, MacKinnon's
  tau_min and tau_max, so p is non-decreasing; plm evaluates the polynomial there (its p falls
  to 0.25 at t = 3 with a trend). A p that underflows is floored at the smallest positive
  double so the statistic stays finite. plm's default with urca installed is
  `p.approx = "MacKinnon1996"` (urca's tables of the 1996 distribution); the two differ by at
  most 0.004 in p on the grid inside the monotone range (ECON-44, decided for the 1994
  surface, which needs no table).
- **Hadri** (`purtest(test = "hadri")`). Model `none` is refused ("without deterministic terms
  the KPSS limit has mean 1/2, not 1/6"), as are unbalanced panels and `lags`, `lag_selection`
  or `pmax` (Hadri has no lagged differences; before, `lags` was ignored silently). Residuals
  `e_i` demeaned or detrended on (1, 1..T); `S_i = sum (cumsum e_i)^2 / T^2`. `hadri_variance`:
  - `heteroskedastic` (default, plm `Hcons = TRUE`): `sigma_i^2 = e'e / (T - nd)` with dfcor,
    `/ T` without; `LM = mean(S_i / sigma_i^2)`;
  - `homoskedastic` (`Hcons = FALSE`): `sigma^2 = sum e'e / (N (T - nd))` (or `N T`);
    `LM = mean(S_i) / sigma^2`;
  - `long_run`: per unit `tseries::kpss.test`'s Bartlett long-run variance with `l` lags
    (default `trunc(4 (T/100)^(1/4))`, kpss.test's `lshort`); dfcor does not apply; `l >= T` is
    an error. Before Tier 4 this was the undisclosed default, with `ceil(T^(1/3))` lags.

  `z = sqrt(N) (LM - mu) / sigma` with `(mu, sigma^2)` = (1/6, 1/45) or (1/15, 11/6300),
  `p = 1 - Phi(z)` (direct upper tail). `unit_statistics` holds `S_i / sigma_i^2` with each
  variant's per-unit variance (plm's `idres$LM` for both short-run variants). The output prints
  `Variance:` and, for the short-run variants, the dfcor line. `hadri_variance` with another
  test is an error; the option is in core only (the MCP and CLI surfaces do not expose it yet).

Conventions: Hadri's short-run variance follows the shared `dfcor` (default true, `rss/(T -
k)`); plm's default is `dfcor = FALSE`. Both are pinned (see ECON-88 above).

### R reference: `purtest_stat_ar05`, `purtest_stat_short`, `mackinnon_punitroot_grid`

`validation/r/cases/econ_purtest_t4.R`, R 4.3.3, plm 2.6.7, urca 1.3.4, tseries 0.10.63.

`purtest_stat_ar05` (`validation/datasets/generated/purtest_ar05.csv`: N = 10, T = 50; `y`
= unit mean i plus a stationary AR(0.5), `y_rw` = random walks):

```r
p <- pdata.frame(data.frame(id, t, y), index = c("id", "t"))
purtest(p$y, test = "levinlin", exo = "intercept", lags = 1, dfcor = FALSE)
# test = "levinlin" | "madwu" (exo intercept | trend | none), "ips" (lags 1 | 3), "hadri"
# (Hcons TRUE | FALSE); dfcor FALSE | TRUE; madwu also with p.approx = "MacKinnon1994";
# kpss.test(x, null = "Level" | "Trend", lshort = TRUE | FALSE) per unit
```

| Quantity | R |
|---|---|
| `llc_intercept_stat` / `_p` | -5.348951494240082 / 4.423262279763195e-08 |
| `llc_intercept_x1000_stat` (1000 y) | -5.348951494240075 |
| `llc_trend_dfcor_stat` | -5.503506788326892 |
| `ips_intercept_dfcor_lags1_stat` / `_p` | -6.269347349485371 / 1.8128226681355234e-10 |
| `madwu_intercept_m94_stat` / `_p` | 94.28427280773165 / 1.3086786422791519e-11 |
| `madwu_intercept_m94_ptrho` (first 3) | 0.004016120733750137, 0.013797702842739526, 0.0006459103293816857 |
| `madwu_trend_m94_stat`, `madwu_none_m94_stat` | 86.91119646185113, 28.453386349780374 |
| `hadri_intercept_dfcor_hcons_stat` | 11.408548635540244 |
| `hadri_intercept_homo_stat`, `hadri_trend_dfcor_hcons_stat` | 12.485146032368666, 9.053573704454731 |
| `kpss_level_short` (first 3), `_lag` | 0.5041592501235846, 0.24340395605727178, 0.06664800734015029; 3 |

`purtest_stat_short` (`purtest_short.csv`: N = 8, T = 15, AR(0.3)-driven random walks; dfcor
TRUE): `ips_intercept_lags4_stat` 1.9186305531704941 (L = 10, an exact row);
`ips_intercept_lags2_moments` -1.3462, 1.2838 (L = 12, interpolated); `llc_intercept_stat`
-1.142455141372953 and `llc_trend_stat` -4.475050634071477 (T = 15, below the table's first
row).

`mackinnon_punitroot_grid` (`mackinnon_grid.csv`: q = -6.5..3 by 0.05): `p94_*` =
`plm:::padf(q, exo, p.approx = "MacKinnon1994")`, `p_nc|c|ct` = `urca::punitroot(q, N = Inf,
trend)` (the deviation). At q = -2.85: p94 0.004277020191323626, 0.05147892440613742,
0.1791864029860865 (none, intercept, trend); urca 0.004254301267439573, 0.05144814729062089,
0.17924142733833812.

Tests (`validation/methods/econ_purtest_t4.json`):
- `econometrics::panel_unit_root::tests::test_validate_mackinnon_p_grid` (method
  `mackinnon_p_values`): the 191 p94 values of each model inside the monotone range
  (`Tol::p_value(191)`); the 53 points beyond the cubic's maximum (6 with an intercept, 47 with
  a trend) equal p at the turning point; p is non-decreasing on the grid.
- Method `panel_unit_root_statistics`: `test_validate_purtest_stat_madwu` (the m94 statistics,
  p and per-unit p for the three models; the per-unit ADF t of both panels, three models, both
  dfcor; the plm-default statistics within 2%, outside the Check),
  `test_validate_purtest_stat_ips` (W-tbar, p and the averaged table moments to 1e-14),
  `test_validate_purtest_stat_llc` (z and p of both panels, three models, both dfcor; the z of
  1000 y; the short case with its table warning), `test_validate_purtest_stat_hadri` (z, p
  and per-unit LM, both variances, both dfcor), `test_validate_purtest_stat_hadri_long_run`
  (the per-unit kpss.test statistics with the short and the long lag, and the lag in the label).

Property tests (no case loaded): `test_mackinnon_p_monotone_with_mackinnons_range` (p
non-decreasing from t = -25 to 5; the turning points are MacKinnon's tau_min, tau_max: -19.04,
-18.83, -16.18; inf, 2.74, 0.70), `test_ips_refuses_model_none`,
`test_ips_untabulated_moments_is_an_error` (T = 18, 5 lags), `test_ips_table_clamp_warnings`,
`test_llc_refuses_unbalanced_panels`, `test_llc_long_run_lag_capped_on_very_short_series`,
`test_llc_unit_sigma_skips_explained_level`, `test_hadri_refuses_model_none_unbalanced_and_lags`,
`test_hadri_variance_option_parses_and_conflicts`, `test_hadri_display_names_variance`.

## Solve and timing

The LLC auxiliary regressions of a unit (Δy and y_{t-1} on the deterministic terms and the
lagged differences) share one design and take OLS's solve decision: one X'X inverse gives both
residual vectors on a full-rank, well-conditioned design (refined once below a reciprocal
condition of 1e-2), and `lm.fit`'s pivoted QR decides rank otherwise, so an exactly linear unit
is still skipped as rank deficient. `test_llc_residuals_normal_equations_match_qr` holds both
residual vectors to 1e-10 of the QR's (relative to the residual norm) on noisy AR(1) units,
with an intercept and with a trend.

Timing: release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds
against `ff8e7559` (before these changes); new / base as the ratio of medians.
`unified` rows use R's `panel_n<n>.csv` (y by entity and time, `floor(T^(1/3))` lags).
Every row is within 1.1x of base.
The outputs the unified runner captures (coefficients, SEs, R², test statistics) equal
base's bit for bit; a pivoted-QR solve differs from them in the last one or two digits (1e-9
relative for the Hausman statistic).

| Fit | n | new / base |
|---|---|---|
| LLC, unified | 100 / 1,000 / 10,000 | 0.78 / 0.84 / 0.80 |

## Tolerances

`validation/README.md`, n = 690: statistics 1e-6 relative; the per-unit taus agree with
`urca::ur.df` to 6e-14. Counts and `avg_t` are exact. `purtest_lag_selection` (n = 480):
lags exact, per-unit t 1e-6 relative (an R transcription agrees with plm to 2e-15).

## Not covered

- The default lag rule and `dfcor` default (ECON-88): with its defaults (`lags = "SIC"`,
  `dfcor = FALSE`) plm's per-unit `trho` differs from the ADF tau pinned above (-3.85 against
  urca's -1.93 for u001); with the same lags and `dfcor = TRUE` they agree
  (`purtest_lag_selection`).
- plm's default per-unit p-values (`p.approx = "MacKinnon1996"`, ECON-44; see "Statistics").
- Refused where plm refuses or returns NA: LLC and Hadri on unbalanced panels, IPS and Hadri
  with model `none`, IPS at untabulated (lags, T); the LLC long-run lag cap for T <= 7 (plm
  has no valid value there).

## MacKinnon (1994) coefficients

The coefficients in `MacKinnon94::of` (`panel_unit_root.rs`) are plm's, printed with
`print(plm:::padf)` (plm 2.6.7): `small` (none, intercept, trend) `0.6344, 1.2378, 3.2496 /
100`, `2.1659, 1.4412, 3.8269 / 100`, `3.2512, 1.6047, 4.9588 / 100`; `large` `0.4797, 9.3557 /
10, -0.6999 / 10, 3.3066 / 100`, `1.7339, 9.3202 / 10, -1.2745 / 10, -1.0368 / 100`, `2.5261,
6.1654 / 10, -3.7956 / 10, -6.0285 / 100`; limits -1.04, -1.61, -2.89 (MacKinnon 1994, Table 3,
N = 1).
