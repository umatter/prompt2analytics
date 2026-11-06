# Multi-cutoff regression discontinuity (`run_rd_multi`)

**Status:** R-pinned (registry `validation/methods/causal_rdbw.json`, id `rd_multi_selected`;
one cutoff at a given bandwidth: `validation/methods/causal_rd.json`, id `rd_multi`). The
per-cutoff estimates are pinned to rdrobust with rdbwselect bandwidths; the pooling is p2a's own
estimator, pinned to its formula written out in R (a registry deviation, see "Relation to
rdmulti::rdmc").

## Method

J cutoffs `c_1 < ... < c_J` share one running variable. Each observation belongs to one cutoff
(`cutoff_col`, or the nearest cutoff, the lowest index on a tie). p2a estimates the effect at
each cutoff, pools the cutoff-specific estimates into one weighted average, and tests whether
they differ.

## What p2a computes

- **Per cutoff:** `run_rd` on the observations of that cutoff (`validation/econometrics/rd.md`):
  rdrobust's bias-corrected estimate with its robust SE, interval at `level` and p-value. With
  `RdMultiBandwidth::PerCutoffOptimal` (the default) h and b are selected by `bwselect` on that
  cutoff's observations (`rdrobust::rdbwselect`, with `kernel`, `p`, `q`, `vce` and
  `scaleregul`), as `rdmulti::rdmc` computes its cutoff-specific estimates. `Global(h)` uses the
  same h at every cutoff and `PerCutoff(hs)` one h per cutoff, with b = h (rdrobust's default
  when h is given). `CutoffResult` reports `h_left/h_right`, `b_left/b_right`, `N_h` and `level`;
  each cutoff's selection warnings are copied into the result as `Cutoff j (c = ...): <text>`.
- **Pooling:** weights `w_j` by effective sample size (`N_h,left + N_h,right`, the default),
  inverse variance (`1/se_j²`) or equal (`1/J`), normalised to sum to 1;
  `tau_pooled = sum_j w_j tau_j` and `se_pooled = sqrt(sum_j w_j² se_j²)` (independent cutoffs).
  The interval is `tau_pooled ∓ z se_pooled` with `z` from `level`, and the p-value is
  `2 Φ(-|tau_pooled / se_pooled|)` (upper tail; a zero SE gives 0 for a nonzero estimate and 1
  for zero, as `run_rd`).
- **Heterogeneity:** `Q = sum_j (tau_j - tau_IV)² / se_j²` with `tau_IV` the inverse-variance
  pooled estimate, and `p = P(chi²(J - 1) > Q)` from the upper tail (a Q of 200 gives
  2.09e-45, not 0).

The printed intervals carry the requested level (`90% CI` at `level = 0.9`).

## Relation to rdmulti::rdmc

`rdmc`'s cutoff-specific estimates are `rdrobust` runs per cutoff with rdbwselect bandwidths,
which p2a reproduces. `rdmc`'s pooled estimate runs rdrobust on the normalised score `x - c` of
all observations; p2a does not implement it. p2a's pooled estimate is the weighted average above,
a different estimator. `rdmulti` is not installed in the validation environment, so the pooling
and the heterogeneity test are pinned to their formulas written out in R in
`validation/r/cases/causal_rdbw.R` (registry deviation CAUS-77; see also CAUS-78).

## Reference cases

Case `rd_bwselect_sharp` (`validation/r/cases/causal_rdbw.R`), keys `multi_*`. The two-cutoff
data are built from `rd_sim.csv`: group 0 is `(y, x)` at c = 0 and group 1 is `(yf, x + 2)` at
c = 2, stacked in that order. `max(x) < 1 < min(x + 2)`, so nearest-cutoff assignment reproduces
the groups (asserted in R). Cutoff 0's values are the `auto_*` keys of `validation/econometrics/rd.md`.

## Decisive values

R 4.3.3, rdrobust 4.0.0.

| key | value |
|---|---|
| cutoff 0 (`auto_*`) | h 0.2641578906068817, b 0.44316662165001586, coef_bc 0.63490809474914, se_robust 0.15747387694470696, p_robust 5.534392633219122e-05, ci_robust [0.3262649674316221, 0.943551222066658], n_h [145, 124] |
| `multi_c2_*` | h 0.39295794749323004, b 0.5951876743491831, coef_bc 0.2451211066131631, se_robust 0.15723791378993096, p_robust 0.11901565015226376, ci_robust [-0.06305954141931541, 0.5533017546456416], n_h [215, 177] |
| `multi_pool_sample` | effect 0.40374848907697214, se 0.11314692592964143, ci [0.18198458929345382, 0.6255123888604904], p 0.0003592284737867214, weights [0.4069591527987897, 0.5930408472012103] |
| `multi_pool_iv` | 0.43972234815546335, 0.1112673268482804, [0.2216423948767872, 0.6578023014341395], 7.751858230792967e-05, [0.49925022503422745, 0.5007497749657726] |
| `multi_pool_equal` | 0.44001460068115156, 0.11126745194912849, [0.22193440220931873, 0.6580947991529844], 7.667336371775572e-05, [0.5, 0.5] |
| `multi_het` | Q 3.068021495314145, p 0.0798465944698513 (df 1) |
| `multi_g03_coef_bc` / `_se_robust` | [0.5919701612168791, 0.4076019486029241] / [0.19836524289095045, 0.21450816491328709] |
| `multi_c2_ci90_robust` | [-0.013512246178487786, 0.503754459404814] |
| `multi_c2_cerrd_h` / `_coef_bc` | 0.27819292236807563 / 0.28786871441346973 |
| `noreg_auto_h` (cutoff 0, scaleregul 0) | 0.5485103365852835 |

Before WP20 every per-cutoff h was 0.49912596890000005 (p2a's former selector, capped at a
quarter of the range).

## Tests and tolerances

Registered:

- `test_validate_rd_multi_single_cutoff` (`rd_multi`, `causal_rd.json`): one cutoff at 0, a
  global bandwidth of 0.3 and HC1 against `rdrobust(y, x, c = 0, h = 0.3, b = 0.3, vce = "hc1")`.
- `test_validate_rd_multi_two_cutoffs_selected` (`rd_multi_selected`): both cutoffs' h, b,
  estimate, SE, p, CI and N_h, the sample-size pooling and the heterogeneity test; the run
  without `cutoff_col` equals the run with the group column.
- `test_validate_rd_multi_pooling_weights`: inverse-variance and equal weights.
- `test_validate_rd_multi_global_bandwidth`: `Global(0.3)` gives h = b = 0.3 at both cutoffs and
  rdrobust's estimates and SEs at that bandwidth.
- `test_validate_rd_multi_level_cerrd_scaleregul`: the 90% interval, `cerrd` and
  `scaleregul = 0`.

Not registered (exact or property tests): `test_pooling_weights_sample_size`,
`test_pooling_weights_inverse_variance`, `test_pooling_weights_equal`, `test_pooled_estimate`,
`test_heterogeneity_test` and `test_heterogeneity_test_homogeneous` (hand-computed values within
1e-15, or 1e-10 relative for the chi-squared tail), `test_assign_to_nearest_cutoff`,
`test_rd_multi_dataset_recovers_dgp` (the DGP effects 1.5 and 2.5 on a simulated two-cutoff
design), `test_rd_multi_interval_label_follows_level`, `test_display_formatting`,
`test_rd_multi_insufficient_data` and `test_rd_multi_validates_parameters_up_front`.

Tolerances (`validation/README.md`, n = 1000): bandwidths and coefficients 1e-8 relative, SEs,
statistics and CI bounds 1e-6 relative, p-values 1e-6 relative plus 1e-12 absolute, counts
exact.

## Missing values and cutoff_col

`run_rd_multi_dataset` reads the outcome, the running variable and `cutoff_col` (when given)
through the shared extractor with the `Omit` policy: a row with a missing value in any of them
is dropped, as rdrobust drops incomplete rows, and reported in `RdMultiResult.missing` and a last
`Missing values: ...` line of the output. Infinite values and text columns are refused by name.
`cutoff_col` must hold whole cutoff indices `0..=J-1` for J cutoffs; any other value is an
error naming the column, the original row and the value (before T2-WP11, 1.7 was read as 1 and
-1 as 0). `run_rd_multi` takes arrays and refuses non-finite values, pointing to
`run_rd_multi_dataset` to drop them. `test_na_rd_multi_dataset_drops_rows_and_checks_cutoff_col`
(not registered; no R value) covers these rules.

## References

- Cattaneo, M. D., Titiunik, R., Vazquez-Bare, G., & Keele, L. (2016). "Interpreting Regression Discontinuity Designs with Multiple Cutoffs". *Journal of Politics*, 78(4), 1229-1248.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2020). "Analysis of Regression Discontinuity Designs with Multiple Cutoffs or Multiple Scores". *Stata Journal*, 20(4), 866-891.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs". *Econometrica*, 82(6), 2295-2326.
- R package `rdmulti`: https://cran.r-project.org/package=rdmulti
