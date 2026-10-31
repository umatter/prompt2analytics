# Validation: Panel unit-root tests (data extraction)

## Method Overview

`run_panel_unit_root` (`crates/p2a-core/src/econometrics/panel_unit_root.rs`) runs the
Levin-Lin-Chu, Im-Pesaran-Shin, Fisher (Maddala-Wu) and Hadri tests on a long-format
panel. Each test is built on per-unit regressions over each unit's series; this document
covers how those series are read from the data and validates the per-unit ADF regression
that the IPS and Fisher tests average or combine.

**Status**: R-pinned for the data extraction and the per-unit ADF t-statistic
(`urca::ur.df`). The LLC, IPS and Fisher statistics are not validated against R (ECON-41,
ECON-43, ECON-44, Tier 4); only the per-unit ADF regression is pinned (urca::ur.df). The
Hadri statistic is not validated either.

## Data requirements and conventions

- **Units** are read as a factor: numeric or text ids. Units are ordered as R's `factor()`
  orders them (numbers numerically, text in byte, C-locale, order); `unit_statistics` and
  `unit_p_values` follow that order. For numeric ids this is the order the extraction used
  before, so results on numeric ids with integer times are unchanged.
- **Time** must hold whole numbers or dates (integers of any width, whole-number floats,
  `Date`); a fractional value is `InvalidSpecification` ("must hold whole numbers").
  Before, times were cast to integers and truncated, so quarter values such as 2000.25
  collided and overwrote each other.
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

Property tests, not registered as R-pinned (no value is compared with R; they load
`validation/datasets/generated/ab_balanced.csv` or `ab_common_gap.csv` without a case):

- `test_pur_fractional_time_is_an_error`: quarter times (2000.25, ...) are refused;
- `test_pur_duplicate_unit_time_is_an_error`: a repeated (unit, time) row is refused;
- `test_pur_short_units_dropped_with_warning`: a unit with 2 observations is dropped and
  named;
- `test_na_pur_int64_and_float64_time`: Int64 and Float64 years give bit-identical
  statistics;
- `test_na_pur_text_variable_is_non_numeric`: a text variable is `NonNumericColumn`.

## Tolerances

`validation/README.md`, n = 690: statistics 1e-6 relative; the per-unit taus agree with
`urca::ur.df` to 6e-14. Counts and `avg_t` are exact.

## Not covered

- The LLC, IPS, Fisher and Hadri statistics and their p-values (ECON-41, ECON-43,
  ECON-44, Tier 4). plm's own per-unit IPS statistic differs from the ADF tau pinned here
  (`purtest`'s `trho` for u001 is -3.85 against urca's -1.93); that is part of ECON-43.
- The ADF residual degrees of freedom and the automatic lag selection (ECON-42, ECON-88,
  Tier 3).
