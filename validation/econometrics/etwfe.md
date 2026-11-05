# Validation: Extended Two-Way Fixed Effects (ETWFE)

**Status: R-pinned, known deviations (CAUS-14: default vcov is unit-clustered CR1; fixest/etwfe
default to iid)** (registry entry `etwfe` in `validation/methods/causal_etwfe.json`; cases
`etwfe_staggered` and `etwfe_unbalanced`).

## Method

`run_etwfe` (`crates/p2a-core/src/econometrics/etwfe.rs`) estimates Wooldridge's extended
two-way fixed effects model as `etwfe::etwfe(fml, tvar, gvar, ivar = unit, cgroup, tref, gref)`
does, and aggregates it as `etwfe::emfx(type)`:

```text
y_it = a_i + l_t + sum_(g,s) D_it 1(G_i = g) 1(t = s) b_gs [+ controls] + e_it
```

- **Cells.** With `cgroup = notyet` (default) one coefficient per treated cohort-period
  (`t >= onset`), none at the reference period `tref` (default: the first period). With
  `cgroup = never` every period of a treated cohort except its reference period `onset - 1`
  (placebo leads included); only never-treated units (and a `gref` cohort) are controls.
- **Reference group.** Never-treated units are coded `first_treat <= 0` or a period after the
  last observed one; all such codes form one reference group. etwfe takes one code (`gref`)
  and does not treat 0 as never treated when periods start at 0; p2a does (the benchmarks use
  0-based periods). `gref` names a cohort that joins the reference group. Without never-treated
  units and without `gref`, `cgroup = notyet` makes the last cohort the reference and drops the
  periods from its onset on, as etwfe does; `cgroup = never` is refused.
- **Anticipation** `d` (library only; etwfe has no such option): cohort `g` is treated from its
  onset `g - d` (did's rule); cohorts keep their `first_treat` label and event keys are
  `t - first_treat`, from `-d`. Pinned as etwfe on `gvar = first_treat - d` (`ant1`). Staggered
  DiD follows did instead (base period `g - 1 - d`, post cells `t >= g`, see `staggered_did.md`),
  so ETWFE's simple and group aggregates include the `d` anticipation periods and did's start
  at `g`.
- **Controls** enter as etwfe's formula controls (`etwfe(y ~ x)`): `x` demeaned within cohort and
  interacted with every cell (`cell:x_dm`), plus `x`, `first_treat::v:x` (treated cohorts) and
  `time::s:x` (periods other than `tref`). Terms the unit effects absorb (a time-invariant
  control's main effect and cohort interactions) are listed in `dropped_terms`, as fixest drops
  them. etwfe's `xvar` (effect heterogeneity by a covariate) is not implemented.
- **Absorption.** Unit and period effects are absorbed exactly, on balanced and unbalanced
  panels: unit means are swept, then the unit-demeaned period dummies are projected off through
  their `(T-1) x (T-1)` normal equations (Frisch-Waugh). No tolerance or iteration is involved.
- **Fit.** The normal equations of the within-transformed columns are built column by column from
  the sparse design (no `n x K` matrix) and solved when well conditioned (OLS's normal-equations
  decision, `regression::ols::normal_inverse_from_gram`). When they are singular or ill
  conditioned, a column-order Cholesky of the equilibrated normal equations removes the control
  columns whose pivot is below 1e-10 of their own within sum of squares (fixest's `collin.tol`,
  made relative to the column; a control's cell interaction in a one-unit cohort is proportional
  to the cell) and the reduced equations decide again. The dense within design goes through the
  pivoted QR (`lm.fit`'s `dqrdc2`, tolerance 1e-7) only as the last resort: a cell that is not
  identified, or equations that stay ill conditioned without collinear controls. A cohort-period
  cell that is absorbed or aliased is an error ("not identified"); collinear control columns are
  removed, listed in `collinear_terms` with a warning, and the model is refitted.

## Standard errors and inference

- `vcov = Cluster` (default): clustered by unit with fixest's CR1 factor
  `G/(G-1) * (n-1)/(n-K)`, `K` = coefficients + periods; cell t tests with `G - 1` df.
- `vcov = Iid`: `SSR / (n - k - rank of the unit and period effects)`; t tests with that df
  (824 on `etwfe_panel`).
- Aggregates (`emfx` simple, event, group, calendar): the mean over emfx's rows (etwfe 0.6.2,
  `post_only = TRUE`, `compress = FALSE`) of the per-row effect gradient `w` (the cell
  coefficient plus `x_dm` times each `cell:x_dm` coefficient), so cohorts are weighted by their
  treated observations; estimate `w'b`, delta-method SE `sqrt(w'Vw)` from the full covariance,
  z p-values, CIs at `confidence_level` (default 0.95). A row without a coefficient (the
  `cgroup = never` reference period, a cell at `tref`) contributes 0; a key whose rows have no
  coefficient (the reference period of the event study) has SE and p-value NaN.

## Reference

`validation/r/cases/causal_etwfe.R`, R 4.3.3, etwfe 0.6.2, fixest 0.14.2, marginaleffects 1.0.0:
`etwfe(y ~ 0 | y ~ xc, tvar = year, gvar = first_treat, data, ivar = id, cgroup, tref, gref,
vcov = ~id | "iid")` and `emfx(type)`. `ivar = id` is passed because etwfe's default
(`ivar = NULL`) absorbs cohort instead of unit effects, which gives the same cells on a balanced
panel but not on an unbalanced one; `vcov` is explicit because fixest 0.14 defaults to iid. The
aggregates are pinned as marginaleffects' values (numerical Jacobian) and as the exact delta
method written out in the case (`.etwfe_delta`); the tests compare with the exact values at the
tight tolerances and with marginaleffects' SEs at 1e-6 relative (they differ by at most 4.7e-7).

| Spec (prefix) | Panel, call | cells | nobs | emfx simple | delta SE | 95% CI |
|---|---|---|---|---|---|---|
| `notyet` | balanced, default | 9 | 960 | 1.563937356319881 | 0.1493020807661981 | 1.2713106552012423, 1.8565640574385194 |
| `notyet_iid` | `vcov = "iid"` | 9 | 960 | 1.563937356319881 | 0.1266790513180721 | 1.3156509781407584, 1.8122237344990033 |
| `never` | `cgroup = "never"` | 21 | 960 | 1.606923430463236 | 0.22261013430336207 | 1.170615584635022, 2.04323127629145 |
| `notyet_xc` | `y ~ xc` | 9 (+16 control terms) | 960 | 1.5900261414941697 | 0.13899852433494203 | 1.3175940398934691, 1.8624582430948702 |
| `tref4` | `tref = 4` | 8 | 960 | 1.2218777919506258 | 0.1386045459456004 | 0.9502178738037221, 1.49353771009753 |
| `ant1` | `gvar = first_treat - 1` | 12 | 960 | 1.2365767789274824 | 0.13859983577858997 | 0.9649260925382803, 1.508227465316685 |
| `unbal` | unbalanced | 9 | 923 | 1.5223815600912378 | 0.14664788125373554 | 1.2349569944248096, 1.809806125757666 |
| `unbal_never` | unbalanced, never | 21 | 923 | 1.455980806965079 | 0.231373349504553 | 1.0024973749537571, 1.9094642389764014 |
| `unbal_xc` | unbalanced, `y ~ xc` | 9 (+16) | 923 | 1.566270139820536 | 0.1366230090928838 | 1.2984939625389955, 1.8340463171020767 |
| `alltreated` | all treated, default (gref 8) | 6 | 840 | 1.6643802977709612 | 0.17084502728617412 | 1.3295301973522973, 1.9992303981896256 |
| `alltreated_gref6` | all treated, `gref = 6` | 2 | 600 | 1.0797298937300186 | 0.19452819899634857 | 0.6984616297197348, 1.4609981577403024 |
| `alltreated_never_gref8` | all treated, never, `gref = 8` | 14 | 960 | 1.6312353773637545 | 0.19997024634190938 | 1.2393008965540098, 2.023169858173499 |

## Tests

R-pinned (registered; n = 600-960: coefficients and aggregate estimates 1e-8 relative, SEs,
p-values and CI bounds 1e-6, counts, df and keys exact, marginaleffects SEs 1e-6):
`test_validate_etwfe_notyet`, `test_validate_etwfe_iid`, `test_validate_etwfe_never`,
`test_validate_etwfe_tref`, `test_validate_etwfe_anticipation`,
`test_validate_etwfe_unbalanced`, `test_validate_etwfe_all_treated`,
`test_validate_etwfe_emfx_simple`, `test_validate_etwfe_emfx_event`,
`test_validate_etwfe_emfx_group`, `test_validate_etwfe_emfx_calendar`,
`test_validate_etwfe_control`, `test_validate_etwfe_cgroup_changes_estimate`,
`test_validate_etwfe_confidence_level`.

Property tests (not registered; they read the case CSVs without loading a case):
`test_etwfe_unidentified_cells_are_an_error` (the all-treated panel with `cgroup = never`,
`gref = 8` and year-2 rows removed: fixest silently drops `.Dtreat:first_treat::6:year::2`;
p2a refuses it, through the QR-aliased and the absorbed path),
`test_etwfe_rejects_degenerate_inputs`, `test_etwfe_two_way_within_matches_dummy_regression`
(exact absorption against a dummy regression, and a disconnected panel),
`test_etwfe_scale_invariant`, `test_etwfe_collinear_controls_are_reported`,
`test_etwfe_structured_fit_matches_dense_qr` (structured normal equations, also with collinear
controls removed at the Gram level, against the dense pivoted QR, 1e-10 relative; asserts which
path each fit took), `test_etwfe_display_reports_what_was_estimated`, and the option
parsers.

DGP tests (not R-pinned; 40 units x 7 periods, cohorts 3/5/never, tolerance 0.02, values from
`etwfe(ivar = uid, vcov = ~uid)` on the fixture): `test_etwfe_cohort_average_recovers_dgp`
(1.99372 and 5.00763 for true 2 and 5), `test_etwfe_event_study_recovers_dgp`,
`test_etwfe_heterogeneous_vs_homogeneous_recovers_dgp` (3.12394 for 3.125, 3.49894 for 3.5),
`test_etwfe_structural_counts`.

## Differences from etwfe

- Unit instead of cohort effects by default (etwfe's `ivar = NULL` absorbs cohort effects).
- The default `vcov` is clustered by unit (CR1, fixest's small-sample factor); fixest 0.14 /
  etwfe 0.6.2 default to iid; `vcov = Iid` reproduces it.
- Never-treated coding: every `first_treat <= 0` or after the last period (above).
- `gref` with never-treated units present: etwfe 0.6.2 turns the never-treated code into a
  treated "cohort 0" with its own cells; p2a keeps never-treated units as controls and adds the
  `gref` cohort to the reference group (not pinned; `gref` is pinned on the all-treated panel).
- Non-identified cells are an error, where fixest drops them silently.
- Controls' within-cohort means: etwfe demeans on its whole `data`, before feols drops rows with
  a missing outcome; p2a omits incomplete rows first (below). Identical when no requested value
  is missing (every pinned case).
- Collinear controls are removed and the model refitted (fixest's rule), not given NaN
  coefficients.
- Not covered: `xvar` heterogeneity (`by_xvar`), non-linear families (`family =`),
  `emfx(window =, post_only = FALSE, compress = TRUE)`.

## Missing values and column types

`run_etwfe` (MCP `etwfe`) reads the outcome, unit, time, treatment, first treatment period and
controls through one `ColumnRequest`:

- a row with a missing value (null or NaN) in any of them is dropped from all of them, as
  fixest and etwfe drop incomplete rows, and `EtwfeResult::missing` and the last Display line
  report it. Nothing is imputed any more: a missing `first_treat` was read as 0 (never treated,
  so a treated unit became a control), a missing time as a period 0, a missing unit id as one
  pooled unit "0" (or ""), a missing treatment or control as 0.0;
- unit ids may be integers, fractional numbers, strings, categoricals or dates; time and
  `first_treat` hold whole numbers or dates (a fractional or infinite value is an error);
  never-treated units are coded 0, and an infinite `first_treat` (a common never-treated code,
  which used to run as a cohort without post cells) is refused with an error that says so;
- a text outcome is `NonNumericColumn`, an absent column `ColumnNotFound`.

Fixture tests, not R-pinned (`crates/p2a-core/src/econometrics/etwfe.rs`):
`test_na_etwfe_null_first_treat_drops_rows`, `test_na_etwfe_null_ids_and_time_drop_rows`,
`test_na_etwfe_errors_name_the_column`.

## Running the tests

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::etwfe
Rscript validation/r/run.R --check etwfe_staggered etwfe_unbalanced
```

## References

- Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and
  Difference-in-Differences Estimators." Working Paper.
- Wooldridge, J.M. (2023). "Simple Approaches to Nonlinear Difference-in-Differences with
  Panel Data." *The Econometrics Journal*, 26(3), C31-C66.
- McDermott, G. `etwfe` R package.
