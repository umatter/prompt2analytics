# Validation: Kruskal-Wallis Rank Sum Test

## Method Overview

The Kruskal-Wallis test is a non-parametric test for comparing medians of two or more independent samples. It is the non-parametric alternative to one-way ANOVA and extends the Mann-Whitney U test to more than two groups.

**Key Parameters:**
- `groups`: Vector of (group_name, values) tuples
- Degrees of freedom: k - 1 (where k = number of groups)

**Test Statistic (H):**
```
H = (12 / (N(N+1))) × Σ(Rj²/nj) - 3(N+1)
```

Where:
- N = total sample size
- nj = sample size in group j
- Rj = sum of ranks in group j

**Tie Correction:**
```
H_corrected = H / (1 - Σ(ti³ - ti)/(N³ - N))
```

Where ti is the number of tied observations in the i-th tie group.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | kruskal.test | R 4.3+ |
| scipy.stats | Python | kruskal | 1.11+ |

## Test Cases

Registry: `validation/methods/stats_rank_t3.json`, ids `kruskal_wallis_ties` and
`fligner_ties_examples` (R 4.3.3, cases of `validation/r/cases/stats_rank_t3.R`). Tolerances: the
`validation/README.md` rows for n < 100 (statistics `Tol::stat`, p `Tol::p_value`); df, ranks,
tie sizes and rank sums exact.

### Test 1: Hollander & Wolfe (1973), p. 116 (the `?kruskal.test` example)

```r
x <- c(2.9, 3.0, 2.5, 2.6, 3.2)   # normal
y <- c(3.8, 2.7, 4.0, 2.4)        # obstructive
z <- c(2.8, 3.4, 3.7, 2.2, 2.0)   # asbestosis
kruskal.test(value ~ group)       # case stat_rank_kruskal_hw
```

| Quantity | R value |
|---|---|
| Kruskal-Wallis chi-squared | 0.7714285714285722 |
| df | 2 |
| p-value | 0.6799647735788936 |
| rank sums | asbestosis 33, normal 36, obstructive 36 |

**Rust test:** `stats::kruskal::tests::test_validate_kruskal_hw`

### Test 2: Hand computation

Groups A = (1, 2), B = (3, 4), C = (5, 6): rank sums 3, 7, 11 and
H = (12 / 42) × (9/2 + 49/2 + 121/2) − 21 = 32/7. Asserted to 1e-12.

**Rust test:** `stats::kruskal::tests::test_kruskal_statistic_hand_computed`

### Test 3: Tie correction by hand

Groups (1, 1, 2, 2, 3), (2, 2, 3, 3, 4), (3, 3, 4, 4, 5): tie groups of sizes 2, 4, 5, 3, 1 in
N = 15, so C = 1 − 210 / 3360 = 0.9375. Asserted to 1e-15.

**Rust test:** `stats::kruskal::tests::test_kruskal_tie_correction_hand_computed`

### Test 4: Ties at any scale (STAT-53)

Case `stat_rank_small_groups` (21 values of order 1e-12 in groups a, b, c of 7, with genuine ties
at 1e-13 spacing; every value lies within 1e-10 of every other). Before, ranks treated values
within 1e-10 as tied: the whole sample became one tie group, the tie correction 0, and the test
returned H = 0, p = 1.

| Quantity | R value | p2a before |
|---|---|---|
| chi-squared (y and y_unit = y × 1e12) | 10.459449404761912 | 0 |
| p-value | 0.005354999318316409 | 1 |
| tie correction | 0.9974025974025974 | 0 |
| rank sums | a 43.5, b 70, c 117.5 | 77, 77, 77 |
| midranks of y | 2, 10, 5, 3.5, 3.5, 7, 12.5, 6, 17.5, 12.5, 8, 14, 1, 11, 15.5, 15.5, 9, 21, 17.5, 20, 19 | all 11 |

**Rust tests:** `stats::ranks::tests::test_validate_midranks_exact_ties_small_scale` (midranks,
tie sizes), `stats::kruskal::tests::test_validate_kruskal_small_scale`

### Test 5: All observations tied

`kruskal.test(list(c(1, 1, 1), c(1, 1)))` gives chi-squared NaN and p NA in R. p2a refuses the
sample (`InvalidSpecification`: "all 5 observations are tied, so the statistic is undefined");
known deviation STAT-53 (the registry records the error in place of R's NaN). Distinct values far
below 1e-10 are not ties: `kruskal.test(list(c(1e-300, 2e-300), 3e-300))` = 1.5 in R and p2a.

**Rust test:** `stats::kruskal::tests::test_kruskal_all_tied_is_error`

### One midrank helper

Kruskal-Wallis, Friedman, Quade, Fligner-Killeen, Mood, Ansari-Bradley and both Wilcoxon tests
rank through `stats::ranks::midranks` (Kruskal-Wallis through `rank_sums_by_group`, the same
ranks summed per group), R's `rank()` (`f64::total_cmp` order, ties by exact
equality, so `-0.0` and `0.0` tie and two distinct doubles never do). The Fligner-Killeen pins
(Fligner's own document is [bartlett.md](bartlett.md)):

| Data | chi-squared | df | p-value |
|---|---|---|---|
| `stat_rank_small_groups`, y ~ g | 0.19930844116601584 | 2 | 0.9051503462897974 |
| "three" (7, 7, 6 values) | 5.102420538669544 | 2 | 0.07798722331725408 |
| "two" (1:5 vs 10 × (1:5)) | 3.330636988639434 | 1 | 0.06800053368477917 |
| InsectSprays (`?fligner.test`) | 14.482781038458608 | 5 | 0.012816779189709193 |

**Rust tests:** `stats::fligner::tests::test_validate_fligner_small_scale`,
`stats::fligner::tests::test_validate_fligner_examples` (case `stat_rank_fligner`);
`stats::fligner::tests::test_fligner_rejects_nan` (a NaN inside a group is an error; the dataset
wrapper drops missing values first, as `fligner.test` does). An infinite value is refused too,
where `fligner.test` returns a value: an infinite group median makes `Inf - Inf` deviations,
which R ranks as NaN.

**Timing.** `fligner_test` refuses a non-finite value in the pass that computes each group's
absolute deviations, and turns ranks into scores with one standard-normal quantile per value
(the argument `(1 + r / (n + 1)) / 2` lies in (0.5, 1), so it needs no check). The statistic is
bit-identical to the version with a separate guard pass and a checked quantile. Release, one P
core, `RAYON_NUM_THREADS=1`, medians of 6 interleaved runs (each a median of 21 samples), four
normal groups with different spreads; base = fe13b889, before the Tier 3 rank changes:

| n | base µs | separate guard, checked quantile µs (ratio) | now µs (ratio) |
|---|---|---|---|
| 100 | 3.06 | 3.42 (1.12) | 3.10 (1.01) |
| 1,000 | 33.8 | 35.6 (1.05) | 32.6 (0.97) |
| 10,000 | 556 | 533 (0.96) | 490 (0.88) |

**Timing (T3-PERF5).** `kruskal_test` needs only the rank sum of each group. It ranked the pooled
sample with `midranks`, which scatters every midrank back to its input position, and then
summed the stored ranks; at n = 100,000 it was 1.19x of base. `rank_sums_by_group` sorts
(key, group) pairs on an integer key with `total_cmp`'s order (`-0.0` and `0.0` share a key)
and lets each tie run add its midrank to its members' groups. Midranks are multiples of 1/2,
so every partial sum is exact below 2^52 and, for n below about 9.5e7 (where
n(n+1)/2 < 2^52), the rank sums, tie sizes, statistic and p-value are bit-identical (`stats::ranks::tests::test_rank_sums_by_group_equals_midranks`, on noisy
and heavily tied data, signed zeros, infinities and tiny distinct values). User-space CPU
cycles per call from the hardware counter (`perf_event_open`; median of 11 batches, one P core,
three interleaved rounds; the machine was shared and loaded, so wall clock was not usable),
three groups as the criterion bench, against `fe13b889`:

| n | 100 | 1,000 | 10,000 | 100,000 |
|---|---|---|---|---|
| new / base | 0.85 | 0.78 | 0.74 | 0.74 |

Wall clock on a quiet machine (2026-11-06, load average 1.2-1.6; release build, one P core, `RAYON_NUM_THREADS=1`, three interleaved rounds against `fe13b889`, medians): the criterion bench gives
0.88 / 0.73 / 0.68 / 0.82 at the same sizes (base 2.5 / 21.3 / 292 / 4,391 µs), the unified
runner 0.89 / 0.73 / 0.66 at n = 100 / 1,000 / 10,000.

## Missing values, labels and the R case

`run_kruskal_test(dataset, value, group)` reads the value and the group in one request through
the shared extractor (`p2a_core::data::columns`):

- **NA policy:** a row whose value or group label is missing is dropped (joint `na.omit`, as
  `kruskal.test(value ~ group)`), and the drop is reported in `missing`. A missing label is
  never a level of its own (before T2-WP07 it became a group named `null`).
- **Labels and order:** groups follow R `factor()` order: text by byte order, numeric labels by
  value (an Int64 group column with labels 2 and 10 gives the groups "2", "10"). Text labels
  are printed as stored, without quotes (STAT-25).
- **Dtypes:** integer values are accepted; a text value column is `NonNumericColumn`, an absent
  column `ColumnNotFound`.

**R case** `stat_oneway_na_labels` (`validation/r/cases/stats_anova.R`, dataset
`validation/datasets/generated/oneway_na_labels.csv`, 9 rows; `g9` is missing in 3 of them):

```r
kruskal.test(v9 ~ g9, data = d)   # 6 rows used: groups a (1, 2, 3) and b (4, 5, 6)
```

| Quantity | R value |
|---|---|
| Kruskal-Wallis chi-squared | 3.8571428571428577 |
| df | 1 |
| p-value | 0.04953461343562679 |
| group sizes | a 3, b 3 |

**Rust tests:**
- `stats::kruskal::tests::test_validate_oneway_na_labels_kruskal` (statistic `Tol::stat(6)`,
  df exact, p `Tol::p_value(6)`, group labels and sizes exact)
- `stats::kruskal::tests::test_validate_na_kruskal_dtypes_and_errors` (Int64 labels in numeric
  order, the text and absent-column errors)
- MCP: `test_validate_mcp_hypothesis_kruskal_wallis_na_labels`
  (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`)

## Numerical Precision Summary

- Statistics and p-values are pinned at the `validation/README.md` rows; ranks, tie sizes, df
  and rank sums are exact.
- P-values are upper tails computed directly (`traits::estimator::chi_squared_p_value`), not
  `1 - cdf`.

## Known Differences

1. **Chi-squared approximation:** Both R and our implementation use the chi-squared approximation, which is accurate when no group has fewer than 5 observations.

2. **Tie handling:** Both implementations rank with exact ties and apply the standard tie
   correction factor.

3. **All observations tied (STAT-53, registry deviation):** p2a returns an error where
   `kruskal.test` returns NaN.

## Performance Comparison

Benchmarked on 2026-01-20.

| Dataset Size | Rust (µs) | R (µs)  | Speedup |
|--------------|-----------|---------|---------|
| n=100        | 9.2       | 2,300   | ~250x   |
| n=1,000      | 49.3      | 6,620   | ~134x   |
| n=10,000     | 1,029     | 37,380  | ~36x    |
| n=100,000    | 10,608    | 492,280 | ~46x    |

**Performance Evaluation:**
- All sizes show Rust significantly faster than R: PASS
- At least 2 sizes >= 2x speedup: PASS (all 4 sizes show 36-250x)
- n=10,000 >= 1.5x speedup: PASS (~36x)

## References

- Kruskal, W. H. & Wallis, W. A. (1952). "Use of Ranks in One-Criterion Variance Analysis". Journal of the American Statistical Association, 47(260), 583-621.
- Hollander, M. & Wolfe, D. A. (1973). *Nonparametric Statistical Methods*. New York: John Wiley & Sons. Pages 115-120.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/kruskal.test.html
