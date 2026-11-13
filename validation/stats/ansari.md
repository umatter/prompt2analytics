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
