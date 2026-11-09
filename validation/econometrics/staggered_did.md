# Validation: Staggered Difference-in-Differences (Callaway-Sant'Anna)

## Method Overview

Staggered DiD estimates group-time average treatment effects ATT(g,t) when treatment timing
varies across units. The Callaway and Sant'Anna (2021) estimator avoids the bias of two-way fixed
effects under heterogeneous treatment effects. p2a's `run_staggered_did` (MCP `staggered_did`,
CLI `causal staggered-did`) computes:

- every cell ATT(g,t) with the universal base period g - 1 - anticipation (`base_period = -1`):
  without covariates the difference in mean outcome changes Y_t - Y_base of cohort g and its
  comparison units, with covariates DRDID's outcome-regression, IPW or doubly robust estimator
  (below);
- the aggregates of `did::aggte`: overall (`type = "simple"`: post-treatment cells weighted by
  P(G = g)), group (`type = "group"`: each cohort's mean over its post-treatment cells) and
  event study (`type = "dynamic"`: for each e = t - g, the cells at e weighted by P(G = g));
- a joint pre-trend test of the pre-treatment cells.

**Key parameters**: `outcome`, `treatment_time` (first treatment period, 0 or negative for
never treated), `time_col`, `unit_col`, `comparison_group` (never treated or not yet treated),
`estimation_method`, `anticipation`, `bootstrap`, `confidence_level`, covariates.

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
`validation/expected/did_cs_analytic.json` (`*_event_*`, `*_cell_*`). The SEs of the base-period
cells and of e = -1 are NA, as did reports them (`se <= sqrt(.Machine$double.eps) * 10` is set to
NA in `att_gt` and `aggte`; CAUS-117); p2a reports NaN, and Display prints `NA`
(`test_validate_staggered_did_reference_cells_have_no_se`).

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
- `test_validate_staggered_did_reference_cells_have_no_se`: the reference cells and e = -1 have
  NaN SEs, analytic and bootstrap, and are left out of the pre-trend test;
- `test_validate_mcp_staggered_did_analytic`, `test_validate_mcp_staggered_did_string_unit_ids`
  (`crates/p2a-mcp/tests/tier2/wp12_did.rs`): the MCP text at print precision
  (`Tol::printed(4)`, the Wald statistic `Tol::printed(2)`).

## Standard errors

- `bootstrap = 0`: analytic standard errors from the cells' unit-level influence functions, as
  `did::att_gt` with `bstrap = FALSE`. The overall and event-study influence functions include
  the term for the estimated weights; the pre-trend test is `a' V^+ a` with the
  influence-function covariance of the pre-treatment cells (base-period cells and cells with a
  zero SE excluded; a singular V uses a generalized inverse with df = rank and a warning). With
  covariates the cell influence functions are DRDID's (below); without covariates the three
  methods coincide, as in did.
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
  below 50.
- With covariates (CAUS-36) the bootstrap is a multiplier bootstrap of the cells' influence
  functions: each cell is fitted once, and a replicate with unit draw counts c_u is
  `att_c + sum_e (c_e - 1) IF_e / n_units` with did's rescaled IF (n_units / n_cell). It uses the
  same random stream as the unit bootstrap and estimates the analytic SE up to Monte Carlo error
  (about 1/sqrt(2B)), as did's default multiplier bootstrap does; re-fitting every cell in every
  replicate would cost B times the fits for the same target. Before, the replicates were the
  unconditional estimator (DR simple SE 0.256 against the analytic 0.170).
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

## Covariates (CAUS-34, CAUS-35, CAUS-37)

Each cell (g, t) is a two-period panel of its treated and comparison units with the outcome
change dY = Y_t - Y_base and the covariates of the earlier period min(t, base) (did's rule; a
time-varying covariate therefore enters at its base-period value in post cells). X = [1,
covariates]. The three DRDID 1.3.0 panel estimators, as `did::att_gt` calls them:

- **Outcome regression** (`reg_did_panel`, `est_method = "reg"`): OLS of dY on X over the
  comparison units, ATT = mean(dY | D = 1) - mean(X b | D = 1), with the influence function of
  the treated mean and the OLS step (A = X0'X0 / n).
- **IPW** (`std_ipw_did_panel`, `"ipw"`): a logit propensity score p(X) (IRLS, `fit_glm`),
  capped at `PS_CAP` = 1 - 1e-6, normalised (Hajek) weights D and p(X)(1 - D)/(1 - p(X)), with the
  propensity-score estimation term in the influence function.
- **Doubly robust** (`drdid_panel`, `"dr"`): the outcome-regression residual dY - X b weighted
  as IPW, one intercept (the old code added a second one and moved with a common trend: CAUS-37).

For IPW and DR, comparison units with p(X) >= `TRIM_LEVEL` = 0.995 get weight 0 (DRDID's
`trim.level`; a warning names the cell).

Guards, as did: a largest propensity score >= 0.999 makes the cell NA ("overlap condition
violated"); collinear covariates (among the comparison units for OR, in the logit for IPW/DR)
make the cell NA ("singular"), not a generalized-inverse fit. A cell that cannot be computed is
dropped with a warning "Could not compute ATT(g,t): ..."; with none left, `InsufficientData`
names the first reason. The base cell (t = base) is ATT 0 with a zero influence function, without
a fit. The point estimate, the analytic SEs and the bootstrap use the same fit of each cell.

### R reference: case `did_cs_covariates`

`validation/r/cases/causal_did_cov.R` writes the three DRDID estimators out from the package
source (the tight reference) and also calls `did::att_gt` / `aggte` 2.5.1 with DRDID 1.4.0
(the `did_*` keys; the script stops when they differ from the transcription by more than 1e-6
relative; measured at most 1.55e-9, not-yet-treated IPW). Panel
`validation/datasets/generated/cs_cov_panel.csv`: 150 units x 8 periods (1200 rows), cohorts 3, 5,
7 and never treated, covariates x1 (continuous), x2 (binary) and x3 (time-varying, shifted by treatment),
cohort membership depending on x1 and x2, covariate-specific trends, and a treatment effect that
starts one period before treatment (anticipation). Blocks: never / not yet
treated, anticipation 0 / 1, OR / IPW / DR, xformla `~ x1 + x2`; `~ x1 + x3` (time-varying,
never treated, anticipation 0) with the period-t covariate rule that the code used before
(`tv_period_t_*`, recorded only); no covariates.

| Quantity (never treated, anticipation 0) | OR | IPW | DR |
|---|---|---|---|
| Overall (simple) ATT | 0.7918185490524854 | 0.7749956313303182 | 0.7918711051051803 |
| Overall SE | 0.16886011892819114 | 0.1697464325298633 | 0.1698903988687777 |

Not yet treated, DR: 0.7333318566434703 (SE 0.1726272959266929). Never treated, anticipation 1,
OR: 2.0000828909247 (SE 0.1632040674478228). ATT(3,3), OR: 0.38619089139987833. Every cell,
aggregate and pre-test value is in `validation/expected/did_cs_covariates.json`.

### Tests and tolerances

R-pinned (`validation/methods/causal_did_cov.json`, method `staggered_did_cs_covariates`), all in
`crates/p2a-core/src/econometrics/staggered_did.rs`; n = 1200 rows: OR ATTs `Tol::coef(1200)`
(1e-10 relative), OR SEs and statistics `Tol::se/stat(1200)` (1e-8); IPW and DR
`Tol::iterative(1e-8)` (the propensity score stops at IRLS epsilon 1e-10 against R's 1e-14;
measured gap 8.8e-13 relative); counts, event times and df exact.

- `test_validate_cs_covariates_or_never`, `test_validate_cs_covariates_or_notyet`: every cell
  ATT, SE and count, the simple, group and event ATT and SE, and the pre-trend W, df and p;
- `test_validate_cs_covariates_or_time_varying`: x3 enters at its earlier-period value (the
  period-t rule differs by more than 0.05 at (3,3));
- `test_validate_cs_methods_coincide_without_covariates`: OR, IPW and DR without covariates are
  the unconditional estimator;
- `test_validate_cs_covariates_ipw_dr_cells`, `test_validate_cs_covariates_ipw_dr_time_varying`;
- `test_validate_cs_covariates_match_did`: the same blocks (anticipation 0 and 1) against did
  2.5.1 itself at `Tol::iterative(1e-6)`;
- `test_validate_cs_anticipation` (below).

Property tests (not registered): `test_cs_dr_invariant_to_common_trend` (adding 10 t to the
outcome changes no DR cell), `test_cs_ipw_dr_refuse_collinear_covariates`,
`test_cs_overlap_guard_skips_cell`, `test_staggered_did_covariate_bootstrap_draws_are_linearised`
(every replicate equals the multiplier draw recomputed from independent fits on T0's stream; the
point estimates do not depend on `bootstrap`), `test_cs_covariates_bootstrap_close_to_analytic`
(the multiplier bootstrap, DR, B = 999, seed 42: overall and group SEs within 10% of the analytic
SEs; a Monte Carlo band, not an R pin, since did's bootstrap draws are not reproduced).

## Anticipation (CAUS-83)

As did 2.5.1 (`pre_process_did`, `compute.att_gt`): with `anticipation = d` the base period is
g + base_period - d (g - 1 - d by default); not-yet-treated comparison units are first treated
after max(t, base) + d; post-treatment cells are t >= g, so the d anticipation cells before
treatment are pre-treatment cells and enter the pre-trend test (method text "(including the d
anticipation period(s) before each cohort's treatment, as did::att_gt)"). A cohort whose base
period is not observed is not computed, with a warning (did drops it). `anticipation` must be
smaller than the number of periods (`InvalidSpecification`). Display prints the anticipation
line. On integer periods the pre-trend W is the same for anticipation 0 and 1
(69.50033372533454 both times, OR, never treated): with the universal base the pre-test is a
re-parametrisation of the same pre-period contrasts. The identification diagnostics do not flag
significant cells inside the configured window (`test_anticipation_window_cells_not_flagged`,
`crates/p2a-core/src/diagnostics/did.rs`).

Tests: `test_validate_cs_anticipation` (never and not yet treated, OR, and never treated DR,
anticipation 1, every cell, aggregate and pre-test value), `test_cs_anticipation_zero_and_one_differ`,
`test_cs_anticipation_out_of_range_is_error`, `test_cs_anticipation_skips_cohort_without_base`.

## No post-treatment cell (CAUS-109)

When no post-treatment ATT(g,t) can be computed (for example every cohort is first treated after
the last observed period), `run_staggered_did` returns `InsufficientData` naming the late cohorts
(did::att_gt treats them as never treated) instead of an overall ATT of 0 with SE 0 and p 1. A
cohort with only pre-treatment cells is left out of the group aggregation with a warning, as
`did::aggte` drops it (`test_cs_no_post_cells_is_error`,
`test_cs_cohort_without_post_cells_left_out_of_groups`).

## Comparison sets

- Never treated: units with `treatment_time <= 0`.
- Not yet treated (`did::compute.att_gt`): never-treated units and units first treated after
  max(t, base) + anticipation, excluding cohort g. Before the fix, cohort g's own units entered the comparison
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
`test_staggered_did_rejects_bad_confidence_level`; MCP: `test_mcp_staggered_did_rejects_small_bootstrap`.

## Not covered

- Varying base periods and the `group`-type overall aggregation (CAUS-84).
- Unbalanced panels and duplicate (unit, time) rows (CAUS-85): did balances the panel first.
- did's recoding of cohorts first treated after the last period as never treated (here: an
  error when no post cell is left, otherwise the cohort has only pre-treatment cells).
- Repeated cross-sections, sampling weights and factor covariates.

## DGP property tests (not R-pinned)

These check properties of known data-generating processes, not R values:
`test_staggered_did_group_time_atts_recover_true_effects_recovers_dgp`,
`test_staggered_did_event_study_pre_treatment_near_zero_recovers_dgp`,
`test_staggered_did_event_study_post_treatment_positive_properties`,
`test_staggered_did_overall_att_weighted_average_properties`,
`test_staggered_did_group_effects_heterogeneity_properties`,
`test_staggered_did_never_treated_comparison_properties`,
`test_staggered_did_homogeneous_effect_recovers_dgp`.

## Running the Tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::staggered_did
cargo test -p p2a-mcp --no-default-features --features "http,llm,auth,websocket" --test tier2 -- wp12_did
Rscript validation/r/run.R --check          # did_cs_analytic and did_cs_covariates still reproduce
```

## References

- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with multiple time
  periods." *Journal of Econometrics*, 225(2), 200-230.
- Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly robust difference-in-differences estimators."
  *Journal of Econometrics*, 219(1), 101-122.
- R package `did` 2.1 (https://bcallaway11.github.io/did/); covariates and anticipation against
  did 2.5.1 and DRDID 1.3.0 / 1.4.0.
