# Validation: Ansari-Bradley Test of Scale (ansari.test)

## Method Overview

The Ansari-Bradley test compares the dispersion of two samples. The pooled sample is ranked
(`stats::ranks::midranks`, R's `rank()`), each observation is scored by its distance from the
ends of the ordering, `min(r, N + 1 - r)`, and the statistic AB is the sum of the scores of the
first sample. p2a reports a normal approximation and, for small samples without ties, an exact
distribution.

## Reference

| Package | Language | Function | Version | Case |
|---------|----------|----------|---------|------|
| stats | R | `ansari.test` | R 4.3.3 | `stat_rank_two_sample`, `stat_rank_small_groups` |

## Method

AB = Σ min(r, N + 1 − r) over the ranks r of the first sample in the pooled sample of size N, the
ranks from `stats::ranks::midranks` (exact ties, as R's `rank()`). The p-value is exact by default
for m, n < 50 without ties (R's rule; `ansari_test_with(x, y, alternative, exact, conf_level)`
overrides it, and exact with ties falls back to the normal approximation with R's warning), else
the normal approximation with R's tie-corrected variance. The exact distribution is R's
`cansari` count, built as a subset-sum recursion over the support `(m+1)²%/%4 ..` with tails
summed directly. As in R, a large AB means a smaller spread of `x`, so `less` (x has the larger
scale) is the upper tail of AB and `greater` the lower tail. Normal tails are computed directly,
not as 1 − cdf. The DP allocates through `resource::try_*` and checks the call's budget once per
outer step.

With `conf_int`, the estimate of the ratio of scales and its interval are R's Bauer (1972)
inversion: exact (the order statistics of the ratios at the exact quantiles, evaluated at the
midpoints as R's `absigma`) when the exact path is taken, otherwise the asymptotic inversion over
the positive ratios |x_i| / |y_j|, found by bisection over their order statistics rather than by
materialising all m n of them, with R's warnings when the samples differ in location or the
interval cannot be computed.
A zero in `x` makes the smallest exact ratios zero, so the exact inversion evaluates AB at a zero
scale, where `0 / 0` is NaN; as R's `rank`, that value ranks last with its own rank and the test
keeps its p-value and interval (`stats::ansari::tests::test_ansari_exact_interval_with_zero_in_x`,
three samples with values quoted from R 4.3.3; not a registered pin).

## Pinned values

Cases `stat_rank_two_sample` and `stat_rank_small_groups` (`validation/r/cases/stats_rank_t3.R`;
datasets `validation/datasets/generated/rank_two_sample.csv` and `rank_small_groups.csv`),
`ansari.test(x, y, exact = FALSE)`; registry `validation/methods/stats_rank_t3.json`, id
`ansari_bradley`. Tolerances: the `validation/README.md` rows for n < 100 (p `Tol::p_value`); AB
exact.

| Set | x, y | AB | two-sided normal p |
|---|---|---|---|
| basic | 1:5, 10 × (1:5) | 15 | 1 |
| scale | 10 values near 5 (5.05 avoids a tie), 10 spread over 0.8-9.2 | 75 | 0.002404179636732229 |
| manual | 1:3, 4:6 | 6 | 1 |
| sim | 20 + 20 normal draws, sd 1 and 3 (seed 20261024) | 251 | 0.026405804321853132 |
| ties | (1, 2, 2, 3, 4), (2, 3, 3, 4, 5) | 14.5 | 0.8537871075965991 (tie-corrected) |
| serum | Hollander & Wolfe serum data (`?ansari.test`) | 185.5 | 0.1814581997286707 (tie-corrected) |
| small scale | `ynt` of groups a and b of `stat_rank_small_groups` (order 1e-12) | 36 | 0.039368435465216844 |

On the tied sets the exact flag is off whatever the configuration asks, as R falls back to the
normal approximation under ties.

**Rust tests:** `stats::ansari::tests::test_validate_ansari_two_sample`,
`stats::ansari::tests::test_validate_ansari_small_scale`

The exact p-values (`ab_p_exact`, `ansari_p_exact`: basic 1, scale 0.0016887137630172688, manual
1, sim 0.02714184174260592, small scale 0.05128205128205132) and the one-sided normal p on
"scale" (`ab_p_less` 0.0012020898183661144, `ab_p_greater` 0.9987979101816339) are pinned too.

## Tier 4 cases (R-pinned)

Case `stat_rank_scale_t4` (`validation/r/cases/stats_rank_t4.R`), registry
`validation/methods/stats_rank_t4.json`, id `ansari_t4`:

- `stats::ansari::tests::test_validate_ansari_exact_t4`: exact p for every alternative (`scale`,
  odd sizes `odd`, 45 + 45 values `ab45`, beyond the old 64-bit overflow), one-sided normal p with
  ties (`ties_p_less`, `ties_p_greater`).
- `stats::ansari::tests::test_validate_ansari_intervals_t4`: exact Bauer interval and estimate
  (`scale`, `cen`, one-sided `cen`), asymptotic interval and estimate on 60 + 60 values (`big`;
  `*_r` is R's `uniroot` root, the unsuffixed value the ratio it approximates).

There are no registry deviations left for the Ansari-Bradley test.

## Input checks

`conf_level` must lie strictly between 0 and 1 (`validate_level`, NaN included) and is checked
before any computation: before, 1.5 or NaN panicked in the normal quantile on the
normal-approximation path. Test: `stats::ansari::tests::test_ansari_rejects_invalid_conf_level`.

## Wall-clock against v0.5.0 (2026-11-13)

Re-timed on 2026-11-13: release builds of v0.5.0 (`f80b0131`) and this version, one P core
(`taskset -c 2`, `RAYON_NUM_THREADS=1`), five rounds interleaving the two builds in alternating order;
each round is one process per build reporting the median of its calls after warm-up calls. Ratios are
of the medians of the five per-round medians, with the ratio of their minima in brackets (1-minute
load average 4.3-8.7 at the start of each round (after the calibration step or the previous round's
conversation-store benchmarks) and 2.2-3.8 at its end; the alternating order protects the ratios).
Two samples of n each. The exact rows are the default call below 50 per sample (R's rule), the
interval rows add `conf_int`: exact (Bauer's inversion) below 50, asymptotic from 100, where v0.5.0
returned no interval, so those base times are the test alone.

| Call | n | v0.5.0 µs | new µs | new / v0.5.0 [min] |
|---|---|---|---|---|
| test, asymptotic | 100 | 2.29 | 2.38 | 1.04 [1.02] |
| test, asymptotic | 1,000 | 29.2 | 29.0 | 0.99 [1.03] |
| test, asymptotic | 10,000 | 496 | 528 | 1.06 [1.07] |
| test, exact | 15 | 25.4 | 26.8 | 1.05 [1.03] |
| test, exact | 30 | 352 | 327 | 0.93 [0.90] |
| test, exact | 49 | 2,612 | 2,608 | 1.00 [1.02] |
| exact interval | 30 | 362 | 673 | **1.86** [1.89] |
| exact interval | 49 | 2,582 | 4,153 | **1.61** [1.56] |
| asymptotic interval (new) | 100 | 2.38 | 1,139 | **478** [516] |
| asymptotic interval (new) | 1,000 | 24.6 | 27,560 | **1,119** [1,159] |
| asymptotic interval (new) | 10,000 | 558 | 821,300 | **1,472** [1,479] |

The interval rows above 1.1x were signed off on 2026-11-13 (rule 5).

On tied data each AB evaluation of the asymptotic interval's root search ranked the pooled sample twice
(once for AB, once for the tie-corrected SIGMA); since 2026-11-20 the squared scores come from the first
ranking, with bit-identical results (`test_ab_normalize_from_one_ranking_matches_ranking_twice`). Timed
on one evaluation (statistic plus z) on tied samples (values rounded to 0.1), n/2 per sample, both
versions in one test binary of the lanes' dev profile (opt-level 1, not a release build), the minimum of
five rounds:

| AB evaluation, tied | n | two rankings µs | one ranking µs | one / two |
|---|---|---|---|---|
| statistic + z | 100 | 8 | 4 | 0.51 |
| statistic + z | 1,000 | 66 | 34 | 0.52 |
| statistic + z | 10,000 | 558 | 290 | 0.52 |
