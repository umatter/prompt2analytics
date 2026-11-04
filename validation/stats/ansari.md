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
ranks from `stats::ranks::midranks` (exact ties, as R's `rank()`). The p-value is the normal
approximation (mean and variance of AB under H0, without a tie adjustment, see STAT-79) or, for
m, n < 50 without ties, an exact distribution (see STAT-13). Normal tails are computed directly
(`normal_p_value_two_sided`, `normal_cdf`), not as 1 − cdf.

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
| ties | (1, 2, 2, 3, 4), (2, 3, 3, 4, 5) | 14.5 | STAT-79 |
| serum | Hollander & Wolfe serum data (`?ansari.test`) | 185.5 | STAT-79 |
| small scale | `ynt` of groups a and b of `stat_rank_small_groups` (order 1e-12) | 36 | 0.039368435465216844 |

On the tied sets the exact flag is off whatever the configuration asks, as R falls back to the
normal approximation under ties.

**Rust tests:** `stats::ansari::tests::test_validate_ansari_two_sample`,
`stats::ansari::tests::test_validate_ansari_small_scale`

## Known differences (registry deviations)

**STAT-13 (exact distribution).** p2a's exact distribution is truncated; R's values stay in the
cases (`ab_p_exact`, `ansari_p_exact`) and are not compared. The counts are `u64` and overflow for
m, n >= 35.

| Set | R exact p | p2a exact p |
|---|---|---|
| basic | 1 | 0.869565 |
| scale | 0.0016887137630172688 | 0.198862 |
| manual | 1 | 1 |
| sim | 0.02714184174260592 | 0.0801778 |
| small scale | 0.05128205128205132 | 0.295762 |

**STAT-12 (one-sided alternatives).** p2a maps `less` to the lower tail of AB and `greater` to the
upper tail; R does the opposite. On "scale" R gives less 0.0012020898183661144 (greater
0.9987979101816339), p2a less 0.998798. R's values: `ab_p_less`, `ab_p_greater`.

**STAT-79 (ties).** The normal approximation ignores ties in the variance: "ties" R
0.8537871075965991, p2a 0.832004; "serum" R 0.1814581997286707, p2a 0.184608 (`ab_p_ties`). AB is
pinned on both.

**STAT-14 (estimate and confidence interval).** Not R's Bauer (1972) inversion; not compared.

## Input checks

`conf_level` must lie strictly between 0 and 1 (`validate_level`, NaN included) and is checked
before any computation: before, 1.5 or NaN panicked in the normal quantile on the
normal-approximation path. Test: `stats::ansari::tests::test_ansari_rejects_invalid_conf_level`.
