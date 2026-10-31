# Validation: Staggered Difference-in-Differences (Callaway-Sant'Anna)

## Method Overview

Staggered DiD estimates group-time average treatment effects ATT(g,t) when treatment timing
varies across units. The Callaway and Sant'Anna (2021) estimator avoids the bias of two-way fixed
effects under heterogeneous treatment effects. p2a's `run_staggered_did` (MCP `staggered_did`,
CLI `causal staggered-did`) computes:

- every cell ATT(g,t) with the universal base period g - 1 (`base_period = -1`), as the
  difference in mean outcome changes Y_t - Y_{g-1} of cohort g and its comparison units;
- the aggregates of `did::aggte`: overall (`type = "simple"`: post-treatment cells weighted by
  P(G = g)), group (`type = "group"`: each cohort's mean over its post-treatment cells) and
  event study (`type = "dynamic"`: for each e = t - g, the cells at e weighted by P(G = g));
- a joint pre-trend test of the pre-treatment cells.

**Key parameters**: `outcome`, `treatment_time` (first treatment period, 0 or negative for
never treated), `time_col`, `unit_col`, `comparison_group` (never treated or not yet treated),
`estimation_method`, `bootstrap`, `confidence_level`.

## R reference

`did` 2.1 is not installed on the reference machine, so `validation/r/cases/causal_did.R`
implements its formulas directly from the package source, case `did_cs_analytic`:

```r
att_gt(yname = "y", tname = "time", idname = "unit", gname = "first_treat",
       control_group = "nevertreated" | "notyettreated", base_period = "universal",
       bstrap = FALSE)          # est_method "reg" without covariates
aggte(.., type = "simple" | "group" | "dynamic")
```

- cell influence functions: `DRDID::reg_did_panel` without covariates, rescaled by n / n_cell;
  SE = sqrt(mean(IF^2) / n);
- aggregation influence functions with the weight-estimation term (`did:::wif`);
- the pre-trend Wald statistic n a' (IF'IF/n)^-1 a over the pre-treatment cells, the base-period
  cells (g, g - 1) excluded.

The panel (`validation/datasets/generated/cs_panel.csv`, written by the case): 90 units x 8
periods (720 rows), cohorts 3, 5, 7 and never treated (`rep(c(3, 5, 7, 0), length.out = 90)`:
23, 23, 22 and 22 units), unit effects, a common trend 0.3 t, a dynamic effect 2 + 0.5 (t - g)
and AR(1) errors (ar = 0.5), seed 20261021. `unit`, `time` and `first_treat` are integers
(Int64 in polars).

The case also records, without comparing them: the `group`-type overall aggregation
(`*_selective_att`, `*_selective_se`; p2a's headline overall ATT is the simple aggregation,
CAUS-84), the diagonal pre-trend form (`*_pretrend_diag_w`, `*_pretrend_diag_p`) and the
independence SE that p2a reported before the fix (`never_simple_se_independent` 0.1087).

### Values

| Quantity | Never treated | Not yet treated |
|---|---|---|
| Overall (simple) ATT | 2.5335019071275213 | 2.593183652513676 |
| Overall SE | 0.20336453729686235 | 0.19751120785805878 |
| Group ATT (g = 3, 5, 7) | 2.868647063638406, 2.084206493389575, 2.421800690431818 | 2.949795004577517, 2.1402323888918477, 2.421800690431818 |
| Group SE | 0.28606411553721534, 0.35392401112910765, 0.3074210190391819 | 0.2651856580359521, 0.3335416145988947, 0.3074210190391819 |
| Event-study SE at e = 0 | 0.16551019275911377 | 0.16397608204964617 |
| Pre-trend Wald W (df 9) | 7.2559128284245595 | 7.236540314019369 |
| Pre-trend p | 0.6104942099687757 | 0.6125070065378957 |

Event-study ATTs and SEs for e = -6 .. 5 and the 24 cell ATTs and SEs are in
`validation/expected/did_cs_analytic.json` (`*_event_*`, `*_cell_*`). The SEs of the four
base-period cells and of e = -1 are exactly 0.

### Tests and tolerances

n = 720 rows (90 units): ATTs `Tol::coef(720)` (1e-8 relative), SEs `Tol::se(720)` (1e-6), CI
bounds and the Wald statistic `Tol::stat(720)` (1e-6), p-values `Tol::p_value(720)`; counts and
degrees of freedom exact. CI bounds and p-values are compared with R literals
(`att -/+ qnorm(0.975) se`, `2 * pnorm(-abs(att / se))`, 1 when the SE is 0).

R-pinned (`validation/methods/causal_did.json`, method `staggered_did_cs`):

- `test_validate_staggered_did_analytic_never`, `test_validate_staggered_did_analytic_notyet`
  (`crates/p2a-core/src/econometrics/staggered_did.rs`): every cell ATT and SE, the cell CIs,
  the overall, group and event-study ATT, SE, CI and p, the pre-trend W, df and p, the pre-trend
  vector, the method text and the cell counts;
- `test_validate_staggered_did_notyet_comparison_sets`: the not-yet-treated comparison counts
  (`G == 0 | (G > max(t, g - 1) & G != g)`: 67, 44 and 22 units) and cell ATTs;
- `test_validate_staggered_did_confidence_level_90`: the 90% CIs (`qnorm(0.95)`) and the
  Display line;
- `test_validate_staggered_did_string_unit_ids`: string unit ids (C order differs from the
  numeric order) and float ids give the integer-id result, pinned to the R ATTs;
- `test_validate_staggered_did_bootstrap_close_to_analytic`: with `bootstrap = 999` (seed 42)
  the overall and group SEs are within 15% of the analytic SEs (observed ratios: 1.019 overall,
  1.063, 0.959 and 1.022 for the groups);
- `test_validate_mcp_staggered_did_analytic`, `test_validate_mcp_staggered_did_string_unit_ids`
  (`crates/p2a-mcp/tests/tier2/wp12_did.rs`): the MCP text at print precision
  (`Tol::printed(4)`, the Wald statistic `Tol::printed(2)`).

## Standard errors

- `bootstrap = 0`: analytic standard errors from the cells' unit-level influence functions, as
  `did::att_gt` with `bstrap = FALSE`. The overall and event-study influence functions include
  the term for the estimated weights; the pre-trend test is `a' V^+ a` with the
  influence-function covariance of the pre-treatment cells (base-period cells and cells with a
  zero SE excluded; a singular V uses a generalized inverse with df = rank and a warning). This
  path is the unconditional estimator: with covariates it is refused (`NotImplemented`, for every
  `estimation_method`; without covariates the three methods coincide, as in did).
- The analytic pre-trend test is did's asymptotic chi-squared test, and it over-rejects on small
  panels: in a null simulation with nine pre-treatment cells it rejected about 19% of panels at
  the 5% level with 90 units (cohorts as small as 15 units) and about 6% with 900, so a
  PARALLEL_TRENDS_VIOLATED warning on a small panel is weak evidence. For small panels use the
  bootstrap test (`bootstrap >= 50`) and its Hotelling F, below.
- `bootstrap >= 50`: a unit bootstrap (the T0 table-based resampler, same random stream). Every
  replicate computes all cells, the aggregates with the replicate's own cohort counts and the
  pre-trend vector; the reported SEs are the SDs of those replicate statistics. A replicate that
  cannot compute a needed cell (no treated or comparison unit drawn, or a drawn unit without its
  base period on an unbalanced panel) is dropped for that statistic; one warning gives the number
  of affected replicates and the fewest any SE used, and recommends `bootstrap = 0` when that is
  below 50. The replicate cell estimator ignores covariates (CAUS-36).
- The bootstrap pre-trend test (CAUS-86) uses the covariance V of the m replicates that computed
  every pre-treatment cell, so `W = a' V^+ a` is Hotelling's T^2: with r = rank V it reports
  `F = (m - r) W / (r (m - 1))` on F(r, m - r) (method text "Hotelling T^2, bootstrap covariance
  of k pre-treatment ATT(g,t) (m of B replicates), F(r, m - r)"; Display "T^2 = .., F(r, m - r) =
  .., p-value = .."). A chi-squared(r) reference ignores the Monte Carlo error in V and
  over-rejects: in the sampling model (a ~ N(0, I_k), V the sample covariance of B independent
  N(0, I_k) draws; 4000 replications in R) the 5% test rejects 15.6% (k = 9, B = 50), 7.0%
  (k = 9, B = 200), 90.7% (k = 30, B = 50) and 18.6% (k = 30, B = 200) of the time, against
  5.1%, 5.2%, 5.4% and 5.1% with the F reference. The test is not computed with fewer than 2
  complete replicates or with m <= r (the identification diagnostics call it untestable).
- `bootstrap` 1..=49 and a `confidence_level` outside (0, 1) are `InvalidSpecification` errors.
  Every CI uses `confidence_level` (statrs normal quantile).
- The joint pre-trend test over k pre-treatment cells costs O(n k^2 + k^3), with n the units
  (analytic) or the replicates used (bootstrap): V is accumulated over blocks of 4096 rows with a
  budget check per block, then eigendecomposed. It is computed for at most 2000 pre-treatment
  cells (`PreTrendTest::MAX_CELLS`); above that it is reported as not computed.

## Comparison sets

- Never treated: units with `treatment_time <= 0`.
- Not yet treated (`did::compute.att_gt`): never-treated units and units first treated after
  max(t, g - 1), excluding cohort g. Before the fix, cohort g's own units entered the comparison
  of its pre-treatment cells and cohorts treated by the base period were comparisons; the
  diagonal pre-trend statistic on the case panel was 68.3 (p 3e-11).

## Missing values and column types

One `ColumnRequest` reads the outcome, `treatment_time`, `time_col`, `unit_col` and the
covariates (in that order): a row with a missing value (null or NaN) in any of them is dropped
from all of them (`NaPolicy::Omit`), and `StaggeredDidResult::missing`, the warnings and the last
Display line ("Missing values: 4 of 720 rows dropped because of missing values (y: 1, ...)")
report it. `did` itself drops units without the needed periods.

- Unit ids may be integers, fractional numbers, strings, categoricals or dates (factor levels in
  R `factor()` order).
- `treatment_time` and `time_col` hold whole numbers or dates; a fractional or infinite value is
  an error naming the column and row. `treatment_time` must be constant within each unit
  (did::att_gt requires it); never-treated units are coded 0 or negative, and an infinite
  `treatment_time` (a common never-treated code) is refused with an error that says so.
- A text outcome is `NonNumericColumn`, an absent column `ColumnNotFound`.
- `n_never_treated` and `n_treated` count units.

Fixture tests (not R-pinned; they compare two Rust runs): `test_na_staggered_did_omits_missing_rows`,
`test_na_staggered_did_errors_name_the_column`. Bootstrap tests:
`test_staggered_did_bootstrap_aggregates_are_replicate_sds` (every SE equals the SD of the
replicate statistic recomputed from the reference replicates, and the pre-trend T^2, F and
p-value equal those of the reference replicates' covariance), `test_pretrend_test` (no rejection
under parallel trends, analytic and with B = 50), `test_pretrend_hotelling_f_matches_pf` (F and
p against R's `pf`), `test_pretrend_blocked_gram_matches_naive_and_checks_every_block`,
`test_staggered_did_bootstrap_reports_dropped_replicates`, `test_staggered_did_bootstrap_ses_pinned`,
`test_bootstrap_draws_match_reference`; argument checks: `test_staggered_did_bootstrap_below_50_is_error`,
`test_staggered_did_analytic_with_covariates_not_implemented`,
`test_staggered_did_rejects_bad_confidence_level`; MCP: `test_mcp_staggered_did_rejects_small_bootstrap`.

## Not covered

- Covariates: the outcome regression ignores them (CAUS-34), IPW/DR covariate rows and the DR
  intercept (CAUS-35, CAUS-37), and the bootstrap replicates are unadjusted (CAUS-36).
- Varying base periods and the `group`-type overall aggregation (CAUS-84).
- Unbalanced panels and duplicate (unit, time) rows (CAUS-85): did balances the panel first.
- Anticipation (CAUS-83).

## DGP property tests (not R-pinned)

These check properties of known data-generating processes, not R values:
`test_validate_staggered_did_group_time_atts_recover_true_effects`,
`test_validate_staggered_did_event_study_pre_treatment_near_zero`,
`test_validate_staggered_did_event_study_post_treatment_positive`,
`test_validate_staggered_did_overall_att_weighted_average`,
`test_validate_staggered_did_group_effects_heterogeneity`,
`test_validate_staggered_did_never_treated_comparison`,
`test_validate_staggered_did_homogeneous_effect`.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::staggered_did
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier2 -- wp12_did
Rscript validation/r/run.R --check          # did_cs_analytic still reproduces
```

## References

- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with multiple time
  periods." *Journal of Econometrics*, 225(2), 200-230.
- Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly robust difference-in-differences estimators."
  *Journal of Econometrics*, 219(1), 101-122.
- R package `did` 2.1 (https://bcallaway11.github.io/did/).
