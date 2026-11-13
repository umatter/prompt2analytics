# Validation: High-Dimensional Fixed Effects (HDFE)

## Method and options

`run_hdfe` absorbs one or more categorical fixed-effect dimensions with the method of
alternating projections (MAP) and runs OLS on the demeaned data, without dummy variables.

- Core: `p2a_core::run_hdfe(dataset, y, x, fe_cols, config, se_type)`; MCP tool:
  `panel_hdfe`.
- `se_type` (`CovarianceType`): `Standard` (iid) or HC0-HC3; HC1 uses `n / df_resid`. There
  is no cluster option.
- `config` (`HdfeConfig`): tolerance (relative, default 1e-8), iteration limit (default
  10000) and acceleration of the MAP; invalid controls and non-finite data are refused.
- Degrees of freedom: `df_resid = n - K - (sum of levels - redundant levels)`.
- Fit statistics: within R-squared, adjusted within R-squared in plm's convention
  `1 - (1 - R2)(n - 1)/df_resid`, and the within F test of the slopes.

## R reference

Case `panel_hdfe_grunfeld`, defined in `validation/r/cases/panel.R`. R 4.3.3, fixest 0.14.2,
plm 2.6.7, on `validation/datasets/grunfeld.csv`:

```r
h <- feols(inv ~ value + capital | firm + year, g, vcov = "iid")
se(summary(h, vcov = "iid")); se(summary(h, vcov = "hetero"))
r2(h, "wr2"); fitstat(h, "wf")
summary(plm(inv ~ value + capital, data = pdata.frame(g, index = c("firm", "year")),
            model = "within", effect = "twoways"))$r.squared["adjrsq"]
feols(inv ~ value + capital | firm, g, vcov = "iid")   # firm effects only
```

## Pinned values (R, 12 significant digits; the case file holds 17)

| quantity | `value` | `capital` |
|---|---|---|
| coefficient (firm + year) | 0.117715855083 | 0.357916273073 |
| SE iid | 0.0137512830036 | 0.0227190108826 |
| SE hetero (HC1) | 0.0191799205213 | 0.0544026642585 |
| t (HC1) | 6.13745270486 | 6.57902104523 |
| p (HC1) | 5.78157933598e-09 | 5.69168652956e-10 |
| SE clustered by firm (not pinned) | 0.0108244294769 | 0.0478483965926 |
| coefficient (firm only) | 0.110123804121 | 0.3100653413 |
| SE iid (firm only) | 0.011856694214 | 0.0173545027756 |

| quantity | value |
|---|---|
| within R² | 0.720145212924 |
| adj. within R² (plm convention, pinned) | 0.670466848355 |
| adj. within R² (fixest `war2`, not pinned) | 0.716833321953 |
| full-model R² (not pinned) | 0.951693399681 |
| within F | 217.442306876 |
| within F df1 | 2 |
| within F df2 | 169 |
| within F p | 1.844236205e-47 |
| df_resid | 169 |
| n | 200 |


## Tests

- R-pinned (`p2a-core`, registered as method `panel_hdfe` in `validation/methods/panel.json`):
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld`: firm + year, iid SEs:
    coefficients, SEs, within R-squared, adjusted within R-squared (plm), within F with its
    df and p-value, `df_resid`, n, and the level counts (10, 20).
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld_hc1`: HC1 SEs, t and p
    against fixest's `hetero`.
  - `econometrics::hdfe::tests::test_validate_hdfe_grunfeld_firm_only`: firm effects only,
    coefficients and iid SEs (the one-way within estimator).
- R-pinned HC0-HC3 (registered as method `hdfe_hc_leverage` in
  `validation/methods/econ_panel_t4.json`, case `hdfe_hc_leverage`):
  `econometrics::hdfe::tests::test_validate_hdfe_hc_leverage_grunfeld` (firm and firm + year),
  `econometrics::hdfe::tests::test_validate_hdfe_hc_leverage_short_panel` (the unbalanced
  short panel, one- and two-way) and
  `econometrics::hdfe::tests::test_validate_hdfe_hc_leverage_disconnected` (two connected
  components): coefficients, HC0, HC1, HC2 and HC3 SEs and `df_resid`. See "HC2 and HC3".
- DGP recovery (no R values compared): `econometrics::hdfe::tests::test_hdfe_coefficients_recovers_dgp`,
  `econometrics::hdfe::tests::test_hdfe_standard_errors_properties`,
  `econometrics::hdfe::tests::test_hdfe_r_squared_properties`,
  `econometrics::hdfe::tests::test_single_fe_against_within_estimator_recovers_dgp`,
  `econometrics::hdfe::tests::test_large_panel_coefficient_recovery_recovers_dgp`. They check
  that known coefficients and degrees of freedom are recovered on simulated panels.

On a balanced firm x year panel the two demeaning projections commute, so the MAP is exact
after one sweep and the Grunfeld pins need no iterative allowance.

## Tolerance

`validation/README.md`, n = 120-300: coefficients 1e-8 relative, standard errors and
statistics 1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; counts and degrees of
freedom exact.

## What changed in Tier 1

`run_hdfe` is unchanged; its tests were not pinned to R. They compared with literals R does
not produce (felm coefficients 0.11772003, 0.35792286 and SEs 0.01375339, 0.02272406, firm-only
0.11013083, 0.31004929; within R-squared 0.7201) at tolerances of 1e-4 to 1e-2, on an
inline copy of the dataset. The page also quoted a felm table for Grunfeld and an R output
for the small simulated example (x1 0.99766, SE 0.02834) that R does not give: fixest on
that data gives x1 0.9802154032 (SE 0.01506135460) and x2 0.4826382350 (SE 0.02363228724)
with 13 residual df. Both are removed.

## Missing values and identifier types

- One NA-aware request reads y, the regressors and every fixed-effect column (the year
  included); the rows with a missing value (null or NaN) in any of them are dropped
  jointly, as plm and fixest do (`na.omit`). HDFE reports the count in `missing` and on the
  last line of its output (it has no FE/RE warning block): "Missing values: 7 of 200 rows
  dropped because of missing values (inv: 2, value: 1, firm_id: 2, year: 2)". The one-way
  FE and RE estimators read no time column; their note, "5 of 200 rows dropped because of
  missing values (inv: 2, value: 1, firm_id: 2)", is also their first warning.
- A missing id drops its row. Before Tier 2 a missing numeric id became 0 and merged
  into the real level 0, and a missing text id became a pseudo-level "" (ECON-37); a
  missing outcome was reported as "Column 'inv' not found".
- Ids may be text or numeric (any integer width, floats, booleans, dates). Levels follow R
  `factor()` order: numbers numerically, text in byte (C locale) order. On Grunfeld and the
  T1 cases this equals the order of first appearance, so their numbers do not change.
- An infinite value is refused (`NonFiniteValues`), a text column in a numeric role is
  `NonNumericColumn`, an unknown column `ColumnNotFound`.

Case `panel_grunfeld_na` (`validation/r/cases/econ_panel_na.R`, R 4.3.3, plm 2.6.7,
fixest 0.14.2): Grunfeld with `inv` missing in rows 5 and 50, `value` in row 120, the firm
id missing in rows 77 and 133 (both as the text id `firm_name`, "F01".."F10", and as the
numeric id `firm_id`, 0..9, so that a real id 0 exists) and `year` missing in rows 31 and
150 (`validation/datasets/generated/grunfeld_na.csv`). FE, RE and the mean-group estimator
take no time column and use the 195 rows complete in (inv, value, capital, id); the two-way
HDFE uses the 193 rows complete in every column.

| Quantity | R (`feols(inv ~ value + capital \| firm_name + year, vcov = "iid")`) |
|---|---|
| coef value, capital | 0.11723697793697486, 0.35306812464135312 |
| iid SE value, capital | 0.01373577849772019, 0.02300641102878161 |
| n used, rows removed | 193, 7 |
| levels firm, year | 10, 20 |

The same model with `firm_id` gives the same partition. On this unbalanced panel the MAP
iterates; an R replica of `demean_map` converges in 7-8 sweeps at the default tolerance to
within 3e-15 of fixest, so the deterministic tolerance applies.

Tests (registered as `panel_missing_values` in `validation/methods/econ_iv_panel_na.json`):
`econometrics::hdfe::tests::test_validate_hdfe_grunfeld_na` (both id columns, the counts,
the note, and the ECON-37 variant with y and x complete),
`econometrics::feglm::tests::test_validate_na_feglm_omit_and_missing_fe_ids`, and through
the MCP tool `tier2::wp08_iv_panel::test_validate_mcp_panel_hdfe_grunfeld_na`. A NaN cell
in y is dropped like a null one, an infinite value refused
(`econometrics::hdfe::tests::test_hdfe_non_finite_data`).

## Convergence of the alternating projections

Each column is demeaned by the method of alternating projections (one sweep subtracts the
group means of every fixed-effect dimension in turn; Gearhart-Koshy acceleration from the
third sweep). A column stops when

    ||change|| <= tolerance * max(||current||, 1e-6 * ||column||)

with `tolerance` the configured 1e-8. The rule is relative: before Tier 3 it was the absolute
`||change|| < 1e-8`, so a column of order 1e9 could never satisfy it and raised
`ConvergenceFailure` after 10000 sweeps, while a column of order 1e-6 stopped after one. The
floor of 1e-6 of the column's own norm keeps a column that the fixed effects absorb (its
demeaned norm tends to 0) from chasing rounding noise. The floor uses the column's uncentred
norm, so it is loose for a column whose mean is large against its variation (values near 1e9
that vary by 1, say): there `1e-6 ||column||` can exceed the demeaned norm, and the sweeps stop
while the demeaned column is still accurate only to that floor rather than to the tolerance. An R transcription of the MAP stops
after 10, 10 and 10 sweeps on `y` scaled by 1e-6, 1 and 1e6 under the relative rule, against
11, 11 and 18 under the absolute one, and agrees with fixest to 1e-15 on both scales.

Reaching the iteration limit is no longer an error: the fit is returned with
`converged = false` and the warning "HDFE alternating projections did not converge after
... iterations ...". `HdfeResult` records `iterations`, `max_iterations`, `tolerance` and the
last relative change (`convergence`); the output prints a `Converged:` line from them.

Case `hdfe_map_scale` (`validation/r/cases/econ_panel_t3.R`, fixest 0.14.2;
`validation/datasets/generated/hdfe_scale.csv`, n = 1000, unbalanced firm x year with 120
absorbed levels): `feols(y ~ x | firm + year, vcov = "iid")` and the same for
`y_big = 1e9 * y`.

| Quantity | R |
|---|---|
| coef x (y) | 0.5412754728 |
| iid SE x (y) | 0.03249037864 |
| coef x (y_big) | 541275472.8 |
| iid SE x (y_big) | 32490378.64 |
| FEGLM Gaussian coef x (y_big) | 541275472.8 |
| within R² | 0.2397677500 |
| df_resid | 880 |
| n | 1000 |

Tests (method `hdfe_map_convergence` in `validation/methods/econ_panel_t3.json`):
`econometrics::hdfe::tests::test_validate_hdfe_converges_at_scale_1e9` (coefficients, iid SEs,
within R², df and n at both scales), `econometrics::hdfe::tests::test_validate_hdfe_map_scale_invariant_iterations`
(the sweep counts at scales 1e-6, 1 and 1e6 differ by at most one),
`econometrics::hdfe::tests::test_validate_hdfe_map_iteration_limit_is_reported` (an
iteration limit of 2 returns the fit with the warning) and
`econometrics::feglm::tests::test_validate_feglm_gaussian_converges_at_scale_1e9`. On this
unbalanced design the MAP stops at its tolerance, so the pins use `Tol::iterative` at the
README values for n = 1000: coefficients 1e-8, SE and R² 1e-6 (the R replica agrees with
fixest to 6.7e-16).

## Regressors absorbed by the fixed effects

FE (`run_fixed_effects`), RE, HDFE and FEGLM screen the demeaned regressors before solving:

1. A regressor whose demeaned sum of squares is at most 1e-12 of its centred sum of squares
   has no variation left after the fixed effects (a time-invariant characteristic under
   entity effects, a column that the entity and time effects together absorb). It is dropped
   with the warning "'z' dropped: no variation left after removing the 'id' fixed effects
   (not estimated, as in plm and fixest)".
2. The remaining demeaned columns take OLS's solve decision (`regression::ols`): when they
   are of full rank and well conditioned (equilibrated reciprocal condition of X'X at least
   1e-8), no column can be aliased and the normal equations solve the fit (refined once
   below 1e-2); otherwise they go through a pivoted QR at `lm.fit`'s tolerance 1e-7, and the
   later of exactly collinear regressors is dropped with "'x3' dropped: collinear with the
   other regressors after removing the 'id' + 't' fixed effects (the later of the collinear
   regressors is not estimated, as in fixest)".

The names are in `dropped`, the warnings in `warnings`; the fit solves from the same
factorization (see "Solve and timing").
With no regressor left the call is an error ("no regressor left to estimate"), as plm's
"empty model". Before Tier 3, FE and HDFE refused such a design with `SingularMatrix` (FE)
or pseudo-inverted it (HDFE).

RE keeps a time-invariant regressor (the between variation identifies it). It drops the
later of collinear columns of the raw design `[1, X]`, as plm's aliased coefficients, and
its within regression, which estimates sigma_e², counts only the slopes that vary within
entities, as plm's `ercomp`. Before Tier 3 that count included the time-invariant regressor,
so sigma_e², theta and every coefficient differed from plm (x1 by 3.6e-3 relative on the case
below). The Hausman test compares the slopes FE keeps.

Case `panel_absorbed_regressors` (`validation/r/cases/econ_panel_t3.R`, plm 2.6.7, fixest
0.14.2; `validation/datasets/generated/panel_absorbed.csv`, N = 30, T = 6): `z` is
time-invariant, `w` is absorbed by the id and t effects together, `x3 = x1 + x2`.

| Model (R call) | Quantity | R |
|---|---|---|
| FE, `plm(y ~ x1 + x2 + z, model = "within")` | coef x1, x2 | 0.1453727769, -0.7468197527 |
| | SE x1, x2 | 0.1742488847, 0.1682010529 |
| | df_resid | 148 |
| RE, `plm(y ~ x1 + x2 + z, model = "random")` | coef (Intercept), x1, x2, z | 0.7365784049, 0.9356803072, -0.5573081128, 0.4100546470 |
| | SE | 0.2074400228, 0.1588304685, 0.1830449165, 0.2043334804 |
| | sigma2 idiosyncratic, individual; theta | 3.617004295, 0.2733851297; 0.1705456109 |
| RE with x3, `plm(y ~ x1 + x2 + x3, model = "random")` | coef (Intercept), x1, x2 | 0.6309324623, 0.6654591694, -0.6321598855 |
| Hausman, `phtest` | chi2, df, p | 120.9620054, 2, 5.412953036e-27 |
| HDFE id + t, `feols(y ~ x1 + x2 + z + w \| id + t)` | coef x1, x2 | 0.4069267592, -0.3027531552 |
| | iid SE x1, x2 | 0.08789712530, 0.08825285307 |
| | df_resid | 143 |
| HDFE id, `feols(y ~ x1 + x2 + z + w \| id)` | coef x1, x2, w | 0.3926827202, -0.3244502460, 1.437028381 |
| | iid SE | 0.08667474941, 0.08522275683, 0.06686447827 |
| FEGLM Gaussian id + t, `feglm(..., gaussian)` | coef x1, x2 | 0.4069267592, -0.3027531552 |

With `x3` added, FE and HDFE drop it and reproduce the coefficients above (`fe_x3_coef`,
`hdfe2_x3_coef`).

Tests (method `panel_absorbed_regressors`):
`econometrics::panel::linear_models::tests::test_validate_fe_drops_time_invariant_regressor`,
`econometrics::panel::linear_models::tests::test_validate_re_absorbed_regressor`,
`econometrics::panel::tests::test_validate_hausman_absorbed_regressor`,
`econometrics::hdfe::tests::test_validate_hdfe_absorbed_regressors`,
`econometrics::feglm::tests::test_validate_feglm_absorbed_regressors`. The design is balanced,
so the MAP is exact after two sweeps and the README tolerances for n = 180 apply
(coefficients 1e-8, SE and statistics 1e-6; names and df exact).

## Disconnected fixed effects

When the firm and year effects split the sample into C connected components, the absorbed
rank is `n_firm + n_year - C`, not `n_firm + n_year - 1`; HDFE counts the components. Case
`hdfe_disconnected` (`validation/datasets/generated/hdfe_disconnected.csv`: 16 firms, 10
years, two components, n = 80) against `lm(y ~ x + z + factor(firm) + factor(year))`:

| Quantity | R |
|---|---|
| coef x, z | 0.6130566690, -0.1963215774 |
| SE x, z | 0.1192968015, 0.1047934716 |
| df_resid | 54 |
| df_absorbed | 24 |
| n | 80 |

With the connected-design count (df_resid 53) the SEs would be 0.1204169838 and
0.1057774695 (`se_naive_df`, not pinned). Test (method `hdfe_disconnected_df`):
`econometrics::hdfe::tests::test_validate_hdfe_disconnected_components`, README tolerances
for n = 80 (coefficients 1e-6, SE 1e-5). FEGLM keeps its own count, which over-counts on
such designs (ECON-85, Tier 4).

## Solve and timing

The regressor screen (`hdfe::screen_regressors`), which FE, RE's within step, HDFE and FEGLM
share, takes OLS's solve decision (`regression::ols::normal_equations_inverse`): on a
full-rank design whose diagonally equilibrated X'X has a reciprocal condition of at least 1e-8
the normal equations solve the fit (refined once below 1e-2) and no column is aliased, because
each unit-scaled column then lies at a distance of at least 1e-4 from the span of the others,
far above the QR's 1e-7; otherwise the pivoted QR at `lm.fit`'s tolerance decides the rank and
the collinear columns, as before. The screen also hands the kept demeaned columns to the
caller instead of a second copy. `test_screen_normal_equations_match_qr` holds the
normal-equations coefficients, (X'X)^{-1} and RSS to 1e-10 relative of the QR's on noisy
two-way designs (k = 2 to 5, a column scaled by 1e4 or 1e-3);
`test_screen_near_collinear_reaches_qr` checks that a column equal to the sum of two others up
to 1e-6 noise reaches the QR and is kept, and that an exactly collinear one is still dropped.

Timing: release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), three interleaved rounds
against `ff8e7559` (before these changes); new / base as the ratio of medians.
`unified` rows use R's `panel_n<n>.csv` (x1, x2, entity effects), `econometrics` rows
criterion's simulated panels.
Every row is within 1.1x of base.
The outputs the unified runner captures (coefficients, SEs, R², test statistics) equal
base's bit for bit; a pivoted-QR solve differs from them in the last one or two digits (1e-9
relative for the Hausman statistic).

| Fit | n | new / base |
|---|---|---|
| HDFE, unified, 2-way | 100 / 1,000 / 10,000 | 0.97 / 0.97 / 0.81 |
| HDFE, econometrics | 100 / 1,000 / 5,000 | 0.91 / 0.95 / 1.04 |

FE, RE and the Hausman test: `panel_fe.md`, `panel_re.md`, `hausman.md`.

## HC2 and HC3: leverage of the fixed effects

HC2 and HC3 weight each squared residual by `1 / (1 - h_i)` or `1 / (1 - h_i)^2`, where `h_i`
is the leverage of the dummy-variable regression, as `sandwich::vcovHC` on
`lm(y ~ x + factor(fe1) [+ factor(fe2)])`. By Frisch-Waugh-Lovell that leverage is
`x̃_i' (X̃'X̃)^-1 x̃_i` of the demeaned regressors plus the diagonal of the projection on the
fixed-effect dummies; before, p2a used the first term only, so HC2/HC3 came out below HC1
(Grunfeld two-way HC3 SE of capital 0.0648 against R's 0.0801).

- One fixed effect: the dummy projection's diagonal is `1 / T_g`, `T_g` the size of the
  observation's level.
- Two fixed effects `a` and `b`, `b` the one with fewer levels: with `S = B' M_A B` (the
  dummies of `b` after the within-`a` projection),
  `h_i = 1 / c_a + (e_b - u_a)' S^+ (e_b - u_a)`, `u_a` the `b`-level shares of `a`'s level.
  `e_b - u_a` lies in the range of `S`, so any generalized inverse gives the same value.
  `S` is singular, with one null vector per connected component of the two effects; p2a
  leaves out one `b`-level per component and inverts the remaining positive-definite block
  by Cholesky, which padded with zeros is a generalized inverse of `S` (the disconnected
  test checks two components). `S` is dense of the order of `b`'s levels, so the cost is
  `O(n_b * cells + n_b^3)`; HC2/HC3
  are computed when the smaller fixed effect has at most 1,000 levels and refused above that
  (`InvalidSpecification`, pointing to HC1).
- Three or more fixed effects: HC2/HC3 are refused, as fixest and reghdfe do not offer them;
  use HC1.
- An observation with leverage 1 (a singleton level, or in a two-way design one that alone
  links two levels) has a zero residual; R's `sandwich` returns NaN for it, fixest drops
  singletons. p2a gives it weight 0 and warns "1 observation has leverage 1 (it is fitted
  exactly, e.g. a singleton fixed-effect level) and contributes nothing to the HC2/HC3 meat".

| design (case `hdfe_hc_leverage`) | HC3 SEs (R) |
|---|---|
| Grunfeld firm | 0.02271635868, 0.05521871232 |
| Grunfeld firm + year | 0.02362378903, 0.08011549708 |
| short one-way (377 rows, 100 ids) | 0.1911267528, 0.1835377347 |
| short two-way | 0.1759447724, 0.1639754516 |
| disconnected (2 components) | 0.1354492321, 0.1429941406 |

Further tests: `econometrics::hdfe::tests::test_hdfe_fe_leverage_matches_dense_projection`
(the leverage against the dense projection on an unbalanced two-way design),
`econometrics::hdfe::tests::test_hdfe_hc2_refuses_three_fixed_effects`,
`econometrics::hdfe::tests::test_hdfe_hc3_refuses_too_many_levels` and
`econometrics::hdfe::tests::test_hdfe_hc3_singleton_group_contributes_nothing`.

### Timing of HC2/HC3

Release builds of v0.5.0 (`f80b0131`) and this version, on one core (`taskset -c 2`,
`RAYON_NUM_THREADS=1`), nine rounds interleaving the two builds; each round reports the median
of at least 41 calls after ten warm-up calls, and the table gives the median of the nine. The
machine was shared with other builds (load average 12-50 on 12 logical CPUs), so ratios
within about 0.05 of 1 are noise. Simulated panels of n / 10 entities over 10 periods with two regressors; the
two-way rows absorb entity and period.

| Fit | n | v0.5.0 µs | new µs | new / v0.5.0 |
|---|---|---|---|---|
| HC1, one-way | 100 | 32.2 | 33.5 | 1.04 |
| HC1, one-way | 1,000 | 173.3 | 194.6 | 1.12 |
| HC1, one-way | 10,000 | 2,758 | 2,848 | 1.03 |
| HC3, one-way | 100 | 42.0 | 36.9 | 0.88 |
| HC3, one-way | 1,000 | 257.3 | 206.0 | 0.80 |
| HC3, one-way | 10,000 | 3,498 | 3,312 | 0.95 |
| HC1, two-way | 100 | 48.7 | 50.0 | 1.03 |
| HC1, two-way | 1,000 | 283.6 | 248.3 | 0.88 |
| HC1, two-way | 10,000 | 3,929 | 3,558 | 0.91 |
| HC3, two-way | 100 | 56.9 | 69.8 | 1.23 |
| HC3, two-way | 1,000 | 343.7 | 405.6 | 1.18 |
| HC3, two-way | 10,000 | 4,739 | 4,237 | 0.89 |

One-way HC3 is faster than before; its fixed-effect leverage is `1 / T_g`. Two-way HC3 at n = 100 and 1,000 costs 1.23x and 1.18x of v0.5.0: v0.5.0 omitted the
fixed-effect leverage and so reported wrong standard errors (see above), and the new cost is
building and inverting the Schur complement of the 10 period levels. This is above the 1.1x
rule and is not signed off. At n = 10,000 the row is 0.89x.

Peak resident memory (`/usr/bin/time -v`), whole process:

| Run | v0.5.0 | new |
|---|---|---|
| data only, 10,000 rows | 6.4 MB | 6.4 MB |
| two-way HC3, 10,000 rows, 10 levels | 6.5 MB | 6.7 MB |
| data only, 20,000 rows, 1,000 x 1,000 levels | 8.1 MB | 8.0 MB |
| two-way HC3, 20,000 rows, 1,000 x 1,000 levels | 12.1 MB | 49.1 MB |

At the 1,000-level ceiling the dense Schur complement, its positive-definite block and the
inverse hold three 1,000 x 1,000 matrices (about 8 MB each); the fit took 0.29 s against 0.03 s
for v0.5.0's incorrect HC3.

### Timing of the one-way demeaning

Columns of a one-way design are demeaned in parallel again when there is more than one
column and enough rows. Same setup, n = 100,000 rows, 10 periods per level:

| Fit | k | threads | v0.5.0 µs | new µs | new / v0.5.0 |
|---|---|---|---|---|---|
| HC1, one-way | 5 | 1 | 97,071 | 82,940 | 0.85 |
| HC1, one-way | 20 | 1 | 598,057 | 553,806 | 0.93 |
| HC1, one-way | 5 | all | 100,289 | 97,186 | 0.97 |
| HC1, one-way | 20 | all | 563,844 | 526,035 | 0.93 |
| FEGLM Poisson, one-way | 5 | 1 | 513,486 | 521,324 | 1.02 |

The multi-threaded rows ran on the shared machine and show little parallel gain there.

## Known differences

- The adjusted within R-squared follows plm (0.6704668484), not fixest's `war2`
  (0.7168333220), which uses a different degrees-of-freedom convention.
- No cluster option, so fixest's firm-clustered SEs are not compared.
- HC2 and HC3 need at most two fixed effects, the smaller with at most 1,000 levels.

## How to run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::hdfe::tests::test_validate
Rscript validation/r/run.R --check
```

## References

- Gaure, S. (2013). lfe: Linear Group Fixed Effects. *The R Journal*, 5(2), 104-117.
- Berge, L. (2018). Efficient estimation of maximum likelihood models with multiple
  fixed-effects: the R package FENmlm. *CREA Discussion Papers* 13 (fixest).
- Grunfeld, Y. (1958). *The Determinants of Corporate Investment*. Ph.D. dissertation,
  University of Chicago.
