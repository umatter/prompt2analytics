# Validation: Panel unit-root tests (data extraction and lag order)

## Method Overview

`run_panel_unit_root` (`crates/p2a-core/src/econometrics/panel_unit_root.rs`) runs the
Levin-Lin-Chu, Im-Pesaran-Shin, Fisher (Maddala-Wu) and Hadri tests on a long-format
panel. Each test is built on per-unit regressions over each unit's series; this document
covers how those series are read from the data and validates the per-unit ADF regression
that the IPS and Fisher tests average or combine.

**Status**: R-pinned for the data extraction, the per-unit ADF t-statistic
(`urca::ur.df`, `plm::purtest`) and the per-unit lag orders of AIC, SIC and Hall selection
(`plm::purtest`). The LLC, IPS and Fisher statistics are not validated against R (ECON-41,
ECON-43, ECON-44, Tier 4). The Hadri statistic is not validated either.

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

Deviations (registry): the IPS W-tbar statistic and p (ECON-43), the LLC statistic and p
(ECON-41), the Maddala-Wu statistic, p and per-unit p (ECON-44), and the defaults (ECON-88):
p2a's default is a fixed `floor(T_max^(1/3))` lags for every unit with `dfcor = true`, while
`plm::purtest` defaults to `lags = "SIC"`, `pmax = 10` and `dfcor = FALSE`. Default-config
outputs are not pinned to plm's defaults; the pins use explicit `lags = "AIC" | "SIC" |
"Hall"`, `pmax = 4` and both `dfcor` values.

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

- The LLC, IPS, Fisher and Hadri statistics and their p-values (ECON-41, ECON-43,
  ECON-44, Tier 4). With its defaults (`lags = "SIC"`, `dfcor = FALSE`) plm's per-unit
  `trho` differs from the ADF tau pinned above (-3.85 against urca's -1.93 for u001); with
  the same lags and `dfcor = TRUE` they agree (`purtest_lag_selection`).
