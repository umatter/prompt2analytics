# Changepoint detection (PELT)

`detect_changepoints` and `run_changepoint` in `crates/p2a-core/src/forecasting/changepoint.rs`.
R reference case: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `lev` of
`validation/datasets/generated/ts_utils.csv`: N(0, 1) x 80, N(0, 3) x 70, N(0, 1) x 60, plus 1e7, 6 decimals.

## Costs and penalties

Costs are those of R's changepoint package, on z = x - mean(x) with compensated prefix sums of z and z^2. For a
segment of length m, with a = sum z and q = sum z^2:

| `CostFunction` | R | Cost |
|---|---|---|
| `VarianceChange` | `cpt.var` (mu = mean of the series) | m (ln 2 pi + ln(q / m) + 1), q <= 0 floored at 1e-11 |
| `MeanAndVariance` | `cpt.meanvar` | m (ln 2 pi + ln s2 + 1), s2 = (q - a^2 / m) / m; at or below tau = 4 eps Q, the rounding level of p2a's centred prefix sums (Q the sum of squares of the whole series), s2 becomes max(tau, 1e-11). R floors s2 = 0 at 1e-11, on raw sums |
| `MeanChange` | `cpt.mean` (unit variance: scale the series first) | q - a^2 / m |

`penalty = None` is changepoint's MBIC: penalty (k + 2) ln n, with k = 1 for a mean or a variance and 2 for both, and
ln(m) added to each segment's cost. `penalty = Some(p)` is "Manual": p and the plain costs. A penalty must be finite and
>= 0, and the data finite (R: "NA is not allowed in the data"). `total_cost` is the sum of the segment costs without the
penalties, so under MBIC it holds the ln(m) terms. The method label names the penalty ("; MBIC penalty", "; manual penalty").

## Pruning

The search returns the optimum that optimal partitioning finds whenever the costs are sub-additive, which holds as
long as no segment's variance sits at the `MeanAndVariance` floor (below). Under a small manual penalty it can stop
above the optimum; see the floor bullet in "Deviations and conventions". Two things differ from changepoint's C code:

1. **Pruning on the base cost.** changepoint prunes s at t when F[s] + C'(s, t) > F[t], with C' holding the ln term. With
   that term the PELT condition (Killick, Fearnhead and Eckley 2012, Theorem 3.1) fails for K = 0: two halves of a
   segment of length L add 2 ln(L / 2), which exceeds ln L. p2a prunes on the cost C without ln. The base costs are
   sub-additive and ln(T - s) >= ln(T - t), so the test is valid. The ln values come from a table.
2. **Deferred drop.** The argument for dropping s at t compares with the segmentation ending at t, which a later T can
   extend only from T >= t + min_seg. A candidate that fails the test stays eligible until then.

Each cost is computed once per (s, t), and candidates are pruned in place. `test_pelt_matches_optimal_partitioning`
compares changepoints and penalised cost with an O(n^2) reference: three seeded noisy series of n = 200, every cost,
MBIC and manual penalties, minimum segments 2 and 7. `test_pelt_matches_optimal_partitioning_with_one_large_value`
covers the MBIC default only (Poisson counts with one value of 1e6 to 1e8).

## Pins

| Test | R call | Pinned |
|---|---|---|
| `test_validate_cpt_var_large_level` | `cpt.var(lev, method = "PELT", penalty = "MBIC")`, and on the centred series | 84, 151; penalty 16.041322592152405; cost 741.4468961546934 = `logLik(cv)[1] + sum(log(seg.len(cv)))` |
| `test_validate_cpt_var_manual_penalty` | `cpt.var(lev, penalty = "Manual", pen.value = 10)` | 84, 151; cost 728.7338492925534 |
| `test_validate_cpt_meanvar_large_level` | `cpt.meanvar(lev - mean(lev), MBIC)` | 2, 84, 152; penalty 21.388430122869874 |
| `test_validate_cpt_mean_large_level` | `cpt.mean(lev - mean(lev), MBIC, minseglen = 2)` | 107, 121, 135, 148, 150; penalty |

Changepoints are compared exactly, the penalty and cost at `Tol::stat(210)`.

`test_pelt_matches_optimal_partitioning`, `test_pelt_matches_optimal_partitioning_on_tied_counts` and
`test_pelt_matches_optimal_partitioning_with_one_large_value` compare the pruned search with a naive reference;
`test_centred_and_raw_costs_agree_on_counts` compares the centred costs with the raw ones on whole numbers. None of the four
compares with R.

## Deviations and conventions

- **TS-18 (recorded).** `cpt.mean` and `cpt.meanvar` work on raw sums: `cpt.meanvar(lev)` reports 54 changepoints at
  level 1e7 (key `cpt_meanvar_cpts`, not compared). p2a centres, so the raw and centred series agree, and the pins call
  R on the centred series.
- **TS-8 (recorded).** `binary_segmentation` has a zero default threshold and no penalty. `cpt.mean(method = "BinSeg",
  penalty = "MBIC", Q = 10)` finds 0 changepoints (key `cpt_mean_binseg_ncpts`, not compared).
- **The floor of `MeanAndVariance` (p2a's own).** While tau < 1e-11, that is Q below about 1.1e4, a tied segment takes
  R's floor. Above that, a tied segment takes tau, so its cost is m ln(tau / 1e-11) above R's: on long count series with a
  large variance the tied segments are rewarded less than in `cpt.meanvar`. One value far from the rest raises tau too:
  ordinary segments under tau take tau, not a floor below it, and are not split off for it
  (`test_meanvar_one_large_value_adds_only_its_own_boundary`: N(0, 1) x 300, N(0, 20) x 300 and a last value of 1e6 or
  1e7 give the regime change and the value's boundary; `cpt.meanvar` gives the same on three seeds, checked by hand).
  **Limit of the pruned search.** A segment whose variance takes the floor breaks the sub-additivity the pruning relies
  on, so the search can then stop above the optimum. That was found only under small manual penalties, in a port of the
  cost and the search (not a run of p2a): N(0, 1) x 200 with one value of 1e7 to 3e8 and manual penalty 1.0 missed the
  optimal-partitioning objective in 57 of 267 series (gaps up to 11), and counts scaled by 1e-6 missed it under manual
  penalties 1 and 10. Manual 8.0 and the MBIC default showed no mismatch in that search, nor did outliers of 1e3 to 1e6
  or counts without an outlier.
- **Limit of centring.** Centring by a mean that one value contaminates leaves the other regimes to rounding. On the same
  series a last value of 1e8 moves the regime change by 5 to 9 positions (305 to 309, three seeds), and 1e9 loses it: p2a
  finds only 598 where `cpt.meanvar` on raw sums finds 300 and 598 at every one of these sizes. Checked by hand against R
  (changepoint 2.3), not pinned, because the positions depend on rounding.
- **Convention: minimum segment 2.** p2a's floor is 2 for every cost; `cpt.mean`'s default `minseglen` is 1. The pin calls
  R with `minseglen = 2`.

## Wall-clock (signed off 2026-11-08)

Medians over five interleaved rounds against `fe13b889` (changepoint.rs there is the tree before this change),
`release`, one P core, `RAYON_NUM_THREADS=1`, on a quiet machine (load average 1.1-1.9, 2026-11-08); MBIC default
penalty, minimum segment 2. Series: `shift`, ten alternating mean regimes (0 and 2); `vshift`, ten alternating
standard deviations (1 and 3); `white`, standard normal noise. Ratio is this tree over the base.

| Row | n | base (ms) | this tree (ms) | ratio (median) | ratio (min) |
|---|---|---:|---:|---:|---:|
| `VarianceChange`, `shift` | 100 | 0.0455 | 0.0447 | 0.98 | 0.98 |
| `VarianceChange`, `shift` | 1,000 | 1.22 | 4.23 | 3.45 | 3.46 |
| `VarianceChange`, `shift` | 10,000 | 73.7 | 423 | 5.74 | 5.76 |
| `VarianceChange`, `vshift` | 100 | 0.0377 | 0.0443 | 1.17 | 1.17 |
| `VarianceChange`, `vshift` | 1,000 | 1.21 | 0.721 | 0.60 | 0.60 |
| `VarianceChange`, `vshift` | 10,000 | 74.2 | 45.9 | 0.62 | 0.62 |
| `VarianceChange`, `white` | 100 | 0.0626 | 0.0467 | 0.75 | 0.75 |
| `VarianceChange`, `white` | 1,000 | 4.53 | 4.43 | 0.98 | 0.98 |
| `VarianceChange`, `white` | 10,000 | 187 | 430 | 2.30 | 2.30 |

The manual penalty 3 ln n was timed only in an earlier round on a loaded machine (4.52x at n = 1,000, 7.39x at
n = 10,000); it was not re-timed.

`MeanChange` and `MeanAndVariance` are faster than the base at every size and series (0.51-0.69). The `VarianceChange` slowdown
follows from the TS-7 cost: measured about the mean of the whole series, it finds no changepoint on these inputs, so
the PELT test never prunes a candidate and the search is quadratic; the old cost about each segment's own mean rewarded
short segments, which is what let it prune. R's `cpt.var` (changepoint 2.3, `minseglen = 2`) with a manual penalty takes
about 990 ms on the same kind of series at n = 10,000, against 1.3-1.4 s here (about 1.4x); with MBIC R takes 4-5 ms
because it prunes on the cost with ln(m), which is not a valid PELT bound (see "Pruning"). Signed off by the user
as a correctness-driven slowdown.
