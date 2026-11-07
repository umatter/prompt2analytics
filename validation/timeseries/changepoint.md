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
| `MeanAndVariance` | `cpt.meanvar` | m (ln 2 pi + ln s2 + 1), s2 = (q - a^2 / m) / m, s2 <= 0 floored at 1e-11 |
| `MeanChange` | `cpt.mean` (unit variance: scale the series first) | q - a^2 / m |

`penalty = None` is changepoint's MBIC: penalty (k + 2) ln n, with k = 1 for a mean or a variance and 2 for both, and
ln(m) added to each segment's cost. `penalty = Some(p)` is "Manual": p and the plain costs. A penalty must be finite and
>= 0, and the data finite (R: "NA is not allowed in the data"). `total_cost` is the sum of the segment costs without the
penalties, so under MBIC it holds the ln(m) terms. The method label names the penalty ("; MBIC penalty", "; manual penalty").

## Exact pruning

The result is the exact optimum, the one optimal partitioning finds. Two things differ from changepoint's C code:

1. **Pruning on the base cost.** changepoint prunes s at t when F[s] + C'(s, t) > F[t], with C' holding the ln term. With
   that term the PELT condition (Killick, Fearnhead and Eckley 2012, Theorem 3.1) fails for K = 0: two halves of a
   segment of length L add 2 ln(L / 2), which exceeds ln L. p2a prunes on the cost C without ln. The base costs are
   sub-additive and ln(T - s) >= ln(T - t), so the test is valid. The ln values come from a table.
2. **Deferred drop.** The argument for dropping s at t compares with the segmentation ending at t, which a later T can
   extend only from T >= t + min_seg. A candidate that fails the test stays eligible until then.

Each cost is computed once per (s, t), and candidates are pruned in place. `test_pelt_matches_optimal_partitioning`
compares changepoints and penalised cost with an O(n^2) reference: three seeded noisy series of n = 200, every cost,
MBIC and manual penalties, minimum segments 2 and 7.

## Pins

| Test | R call | Pinned |
|---|---|---|
| `test_validate_cpt_var_large_level` | `cpt.var(lev, method = "PELT", penalty = "MBIC")`, and on the centred series | 84, 151; penalty 16.041322592152405; cost 741.4468961546934 = `logLik(cv)[1] + sum(log(seg.len(cv)))` |
| `test_validate_cpt_var_manual_penalty` | `cpt.var(lev, penalty = "Manual", pen.value = 10)` | 84, 151; cost 728.7338492925534 |
| `test_validate_cpt_meanvar_large_level` | `cpt.meanvar(lev - mean(lev), MBIC)` | 2, 84, 152; penalty 21.388430122869874 |
| `test_validate_cpt_mean_large_level` | `cpt.mean(lev - mean(lev), MBIC, minseglen = 2)` | 107, 121, 135, 148, 150; penalty |

Changepoints are compared exactly, the penalty and cost at `Tol::stat(210)`.

## Deviations and conventions

- **TS-18 (recorded).** `cpt.mean` and `cpt.meanvar` work on raw sums: `cpt.meanvar(lev)` reports 54 changepoints at
  level 1e7 (key `cpt_meanvar_cpts`, not compared). p2a centres, so the raw and centred series agree, and the pins call
  R on the centred series.
- **TS-8 (recorded).** `binary_segmentation` has a zero default threshold and no penalty. `cpt.mean(method = "BinSeg",
  penalty = "MBIC", Q = 10)` finds 0 changepoints (key `cpt_mean_binseg_ncpts`, not compared).
- **Convention: minimum segment 2.** p2a's floor is 2 for every cost; `cpt.mean`'s default `minseglen` is 1. The pin calls
  R with `minseglen = 2`.
