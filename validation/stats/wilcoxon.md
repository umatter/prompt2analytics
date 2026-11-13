# Validation: Wilcoxon Rank Sum and Signed Rank Tests

## Method Overview

This document validates the p2a-core implementation of Wilcoxon non-parametric tests against R's `stats::wilcox.test()`.

**Functions Implemented:**
- `wilcoxon_rank_sum()` - Mann-Whitney U / Wilcoxon rank sum test for two independent samples
- `wilcoxon_signed_rank()` - Wilcoxon signed rank test for paired samples or one-sample median test
- `wilcoxon_test()` - Dataset wrapper for both tests

**Key Parameters:**
- `alternative` - Direction of alternative hypothesis (two-sided, greater, less)
- `mu` - Hypothesized location shift (default: 0)
- `paired` - Whether to perform paired test
- `exact` - Whether to compute exact p-value (auto-decides if not specified)
- `correct` - Whether to apply continuity correction (default: true)
- `conf_int` - Whether to compute confidence interval and location estimate

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `wilcox.test()` | R 4.3+ |
| scipy | Python | `wilcoxon()`, `mannwhitneyu()` | 1.11+ |

## Mathematical Formulas

### Wilcoxon Rank Sum (Mann-Whitney U)

For two independent samples X (n₁ observations) and Y (n₂ observations):

```
1. Pool samples and rank all values 1 to N = n₁ + n₂
2. W = sum of ranks in sample X
3. U = n₁n₂ + n₁(n₁+1)/2 - W

Under H₀ (same distribution):
E(W) = n₁(N+1) / 2
Var(W) = n₁n₂(N+1) / 12

Normal approximation with continuity correction:
z = (W - E(W) ± 0.5) / √Var(W)
```

### Wilcoxon Signed Rank

For paired samples or one-sample median test:

```
1. Compute differences: dᵢ = xᵢ - μ (or xᵢ - yᵢ)
2. Remove zero differences
3. Rank absolute differences |dᵢ|
4. V = sum of ranks where dᵢ > 0

Under H₀ (median = μ or no difference):
E(V) = n(n+1) / 4
Var(V) = n(n+1)(2n+1) / 24

Normal approximation with continuity correction:
z = (V - E(V) ± 0.5) / √Var(V)
```

## Test Cases

All values in Tests 1-6 are R 4.3.3 from cases `stat_rank_wilcoxon` and `stat_rank_small_groups`
(`validation/r/cases/stats_rank_t3.R`; datasets `validation/datasets/generated/rank_wilcoxon.csv`
and `rank_small_groups.csv`), registry `validation/methods/stats_rank_t3.json`, id `wilcoxon_t3`.
Tolerances: the `validation/README.md` rows for n < 100 (p `Tol::p_value`); statistics exact.
The rank-sum `statistic` is R's W = R1 − n₁(n₁ + 1)/2 (R1 the rank sum of `x - mu` in the pooled
sample `c(x - mu, y)`), `u_statistic` is the same W (the Mann-Whitney U of `x`), and R1 is reported
as `rank_sum`; the signed-rank `statistic` is V. Display labels them `W` and `V`, as R.

### Test 1: Two-sample rank sum (x13a)

```r
wilcox.test(c(1.2, 2.3, 3.1, 4.5, 5.2), c(2.1, 3.4, 4.2, 5.8, 6.1, 7.2), exact = FALSE, correct = TRUE)
```

R: W = 8 (rank sum 23), p = 0.23533326650958575. Pinned.

### Test 2: One-sample signed rank (x13b)

```r
wilcox.test(c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30), mu = 1.5, exact = FALSE, correct = TRUE)
```

R: V = 33, p = 0.23613697022165284. Pinned.

The previous version of Tests 1 and 2 quoted "W = 7, p 0.082" and "V = 40, p 0.146", which R does
not print, and accepted p-value ranges.

### Test 3: Exact test, small sample (exact3)

`wilcox.test(1:3, 4:6, exact = TRUE)`: W = 0 (rank sum 6), p = 0.1 (= 2/20). Pinned.

### Test 4: Paired signed rank with a zero and tied |d| (paired)

```r
x <- c(125, 115, 130, 140, 140, 115, 140, 125, 140, 135)
y <- c(110, 122, 125, 120, 140, 124, 123, 137, 135, 145)
wilcox.test(x, y, paired = TRUE, exact = FALSE, correct = TRUE)
```

R: V = 27, p = 0.6352893188352069. Both pinned; the tie correction of the variance is
Σ(t³ − t)/48 over the tie groups of |d|, as R.

**Rust test (Tests 1-4):** `stats::wilcoxon::tests::test_validate_wilcoxon_examples`

### Test 5: N = 30 and n = 25: R's default exact test

Rank sum on 15 + 15 tie-free values (set n30) and signed rank on 25 tie-free values without zeros
(set sr25). The default configuration follows R's rule (exact when each sample has fewer than 50
values, signed rank when fewer than 50 non-zero differences, and no ties or zeros) and reports
`exact = true`; `exact: Some(false)` gives the normal approximation with continuity correction.

| Set | R default (exact) | R (exact = FALSE, correct = TRUE) | pinned |
|---|---|---|---|
| n30 | W 51 (rank sum 171), p 0.009874822650594207 | p 0.011400982911555176 | W, rank sum, both p, exact and continuity flags |
| sr25 | V 231, p 0.06670159101486205 | p 0.06729929952386755 | V, both p, exact and continuity flags |

**Rust test:** `stats::wilcoxon::tests::test_validate_wilcoxon_n30_exact_flag`

### Test 6: Values of order 1e-12

Rank sum on the tie-free column `ynt` of groups a and b of `stat_rank_small_groups` (7 + 7, all
within 1e-10 of each other), exact: W = 32 (rank sum 60), p = 0.3828671328671329.

**Rust test:** `stats::wilcoxon::tests::test_validate_wilcoxon_small_scale`

### Input checks

`conf_level` is validated when `conf_int` is set (outside (0, 1) or NaN is an error naming
`conf_level`, as R's "'conf.level' must be a single number between 0 and 1"; before, 1.5 or NaN
panicked in the normal quantile), and a non-finite `mu` is an error naming `mu` (R: "'mu' must be
a single number"). Test: `stats::wilcoxon::tests::test_wilcoxon_rejects_invalid_conf_level_and_mu`.

## Missing values and the R case

`wilcoxon_test(dataset, x, y, mu, alternative, paired, config)` reads its columns through the
shared extractor (`p2a_core::data::columns`), with `wilcox.test`'s NA handling:

- **Paired test:** a pair with a missing `x` or `y` is dropped (`complete.cases(x, y)`).
- **Rank-sum test:** each sample drops its own missing values (two independent columns); the
  two summaries are merged (`MissingSummary::merge_samples`), so `missing.n_total` counts the
  values of both samples, twice the rows. The Display line counts values too
  (`MissingSummary::two_sample_note`): `Missing values: 2 of 20 values dropped because of
  missing values (two samples of 10; x: 1, y: 1)`.
- **One-sample test:** the column's missing values are dropped.

The drop is reported in `missing` and in a last Display line `Missing values: ...`. Integer
columns are accepted; a text column is `NonNumericColumn`, an absent one `ColumnNotFound`. Before
T2-WP07 a missing cell was read as 0 and entered the statistic.

**R case** `stat_wilcoxon_na` (`validation/r/cases/stats_anova.R`, dataset
`validation/datasets/generated/wilcoxon_na.csv`, 10 rows; `x` is missing in row 3 and `y` in
row 8, so 8 complete pairs and 9 values per sample; the case checks that there are no ties):

```r
wilcox.test(d$x, d$y, paired = TRUE, exact = FALSE, correct = TRUE)
wilcox.test(d$x, d$y, paired = TRUE)                       # exact
wilcox.test(d$x, d$y, exact = FALSE, correct = TRUE)       # rank sum
```

| Quantity | R value | Quantity | R value |
|---|---|---|---|
| paired V | 35 | rank sum of x (R1) | 100 |
| paired p (normal, corrected) | 0.020862582332765507 | R's W = R1 - 45 | 55 |
| paired z | 2.310462138646216 | rank-sum p (normal, corrected) | 0.21637300805763746 |
| paired p (exact) | 0.015625000000000003 | rank-sum z | 1.236230199927374 |
| pairs used | 8 | n_x / n_y | 9 / 9 |

The z values are recovered from R's p (`-qnorm(p / 2)` with the sign of the statistic minus its
mean).

The rank-sum test pins R's W (`statistic`) and R1 (`rank_sum`).

**Rust tests** (`crates/p2a-core/src/stats/wilcoxon.rs`):
- `stats::wilcoxon::tests::test_validate_wilcoxon_na_paired` (V and the pair count exact; p
  `Tol::p_value(8)`, z `Tol::stat(8)`; the exact p; the missing-value summary)
- `stats::wilcoxon::tests::test_validate_wilcoxon_na_rank_sum` (R1 and W exact, p and z at
  n = 18, sample sizes; 2 of 20 values dropped)
- `stats::wilcoxon::tests::test_validate_na_wilcoxon_dtypes_and_errors` (on the Int64 case
  `stat_friedman_quade`: the one-sample test on an Int64 column equals the test on its Float64
  cast; a text column is `NonNumericColumn`, an absent one `ColumnNotFound`)
- MCP: `test_validate_mcp_hypothesis_wilcoxon_paired_na` (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`)

## Numerical Precision Summary

- Statistics (W, R1, V) are exact; p-values, bounds and estimates are pinned at the
  `validation/README.md` rows.
- Ranks come from `stats::ranks::midranks` (R's `rank()`, exact ties); normal tails are computed
  directly (no `1 - cdf`) and quantiles from `traits::estimator::normal_quantile`.

## Tier 4 cases (R-pinned)

Case `stat_rank_wilcoxon_t4` (`validation/r/cases/stats_rank_t4.R`, datasets
`validation/datasets/generated/rank_*_t4.csv`), registry `validation/methods/stats_rank_t4.json`,
ids `wilcoxon_t4` and `pairwise_wilcox_ties_t4`.

| Test | Sets and quantities |
|---|---|
| `stats::wilcoxon::tests::test_validate_wilcoxon_rank_sum_t4` | `depr` at mu = 0.5 and 0, `tf` at mu = 0.3 (W, p for every alternative), `lik` and `heavy` with ties (W, p with and without continuity correction, z), `n4530` (R's default exact p: each sample below 50), `n5510` (normal: one sample of 55), `n60` (exact requested) |
| `stats::wilcoxon::tests::test_validate_wilcoxon_signed_rank_t4` | `lik` paired with ties (V, p, non-zero count), `srt`, `sr40` (exact, less), `sr60` (exact requested), `srz` (zeros: normal path with R's warning; exact requested with `correct = FALSE`), `sr12` |
| `stats::wilcoxon::tests::test_validate_wilcoxon_intervals_t4` | exact intervals and estimates (`tf` at levels 0.95 and 0.9 and one-sided, `sr40`), asymptotic intervals with ties (`depr`, `lik`, `srt`, `sr12`), with and without continuity correction |
| `stats::pairwise::tests::test_validate_pairwise_wilcox_ties_t4` | `pairwise.wilcox.test` on three tied groups: p unadjusted and Holm |

**Exact distributions.** The rank-sum null distribution is counted exactly (the coefficients of the
Gaussian binomial, in 128-bit integers; above that range, which R's default never reaches, a
probability recursion in f64); the signed-rank distribution by the halving recursion of R's
`csignrank`. Both allocate through `resource::try_*` and check the call's budget once per factor.
Tails are summed directly. Exact is used only without ties (and, for V, without zeros); a request
for exact with ties or zeros falls back to the normal approximation with R's warning ("cannot
compute exact p-value with ties" / "with zeroes").

**Intervals and estimates.** Exact intervals are R's order statistics of the pairwise differences
`x_i - y_j` (rank sum) or the Walsh averages (signed rank) at `qwilcox`/`qsignrank`, selected
without materialising all m n (or n(n+1)/2) values; the estimate is their median. Asymptotic
intervals invert the normal approximation as R does with `uniroot` on the step function; p2a
returns the jump point of that step function, which R's root approximates to its tolerance
(the case records both: `*_r` is R's printed root, the unsuffixed value the jump point it lies
on). Where the step function is flat at zero the estimate is the midpoint of the flat interval
(R's root lands inside it). The signed-rank asymptotic interval doubles alpha as R does when the
requested level cannot be reached, with R's warning; an unreachable exact level returns R's
"requested conf.level not achievable" warning and the achieved level.

## Known Differences

1. **Continuity correction:** both apply the ±0.5 correction by default in the normal
   approximation (`sign(z) * 0.5` two-sided).

2. **Asymptotic bounds:** reported at the jump point rather than at R's `uniroot` approximation of
   it; the difference is below R's root tolerance (1e-4 of the data range), and the case pins the
   jump points.

## Performance Comparison

*Rust benchmark results from `cargo bench -p p2a-core -- Wilcoxon` on 2026-01-19*

### Rank Sum Test (two independent samples, each n)

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=10         | 0.81      | ~30     | ~37x    |
| n=50         | 4.2       | ~50     | ~12x    |
| n=100        | 8.0       | ~70     | ~9x     |
| n=500        | 45        | ~150    | ~3x     |
| n=1,000      | 100       | ~300    | ~3x     |

### Signed Rank Test (paired samples)

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=10         | 0.65      | ~25     | ~38x    |
| n=50         | 1.7       | ~40     | ~24x    |
| n=100        | 4.8       | ~60     | ~12x    |
| n=500        | 26        | ~120    | ~5x     |
| n=1,000      | 46        | ~200    | ~4x     |

### Exact Test (small samples, no ties)

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=5          | 6.4       | ~20     | ~3x     |
| n=10         | 84        | ~50     | ~0.6x** |
| n=15         | 1.2       | ~150    | ~125x   |
| n=20         | 2.0       | ~500    | ~250x   |

*Note: R timings approximate from `system.time()` resolution; microbenchmark recommended.*
**Note: these timings predate the exact distributions of Tier 4 (exact now runs for each sample
below 50, as R). The current comparison with v0.5.0 follows.

### Against v0.5.0 (2026-11-13)

Re-timed on a quiet machine on 2026-11-13: release builds of v0.5.0 (`f80b0131`) and this version,
one P core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), five rounds interleaving the two builds in
alternating order; each round is one process per build reporting the median of its calls after
warm-up calls. The table gives the ratio of the medians of the five per-round medians, with the
ratio of their minima in brackets (1-minute load average 1.7-2.6). Two samples of n each (signed
rank: n pairs); the exact rows are the default call at sizes where R's rule picks the exact
distribution, which v0.5.0 approximated, so they are slower by construction; the interval rows
select the order statistics they need (see above) where v0.5.0 built every pairwise difference or
Walsh average.

| Call | n | v0.5.0 µs | new µs | new / v0.5.0 [min] |
|---|---|---|---|---|
| rank sum, asymptotic | 100 | 2.56 | 2.56 | 1.00 [1.07] |
| rank sum, asymptotic | 1,000 | 29.0 | 28.5 | 0.99 [1.05] |
| rank sum, asymptotic | 10,000 | 538 | 517 | 0.96 [0.95] |
| rank sum, ties | 100 | 2.28 | 2.34 | 1.03 [1.00] |
| rank sum, ties | 1,000 | 17.9 | 16.5 | 0.92 [0.99] |
| rank sum, ties | 10,000 | 148 | 145 | 0.98 [0.98] |
| rank sum, exact | 15 | 0.56 | 3.12 | 5.57 [5.59] |
| rank sum, exact | 30 | 0.94 | 24.9 | 26.53 [29.42] |
| rank sum, exact | 49 | 1.44 | 90.9 | 63.11 [64.29] |
| rank sum with interval | 100 | 308 | 36.8 | 0.12 [0.12] |
| rank sum with interval | 1,000 | 49,524 | 738 | 0.01 [0.02] |
| rank sum with interval | 10,000 | 6,982,718 | 19,419 | 0.00 [0.00] |
| signed rank, asymptotic | 100 | 1.49 | 1.61 | 1.08 [1.07] |
| signed rank, asymptotic | 1,000 | 17.3 | 15.9 | 0.92 [1.06] |
| signed rank, asymptotic | 10,000 | 267 | 242 | 0.91 [0.92] |
| signed rank, exact | 25 | 0.56 | 2.20 | 3.93 [4.26] |
| signed rank, exact | 40 | 0.72 | 5.91 | 8.21 [8.62] |
| signed rank, exact | 49 | 1.11 | 12.1 | 10.90 [13.08] |
| signed rank with interval | 100 | 126 | 23.2 | 0.18 [0.17] |
| signed rank with interval | 1,000 | 23,063 | 449 | 0.02 [0.02] |
| signed rank with interval | 10,000 | 3,344,119 | 11,452 | 0.00 [0.00] |
| pairwise, asymptotic | 100 | 16.1 | 16.6 | 1.04 [1.05] |
| pairwise, asymptotic | 1,000 | 186 | 187 | 1.01 [1.03] |
| pairwise, asymptotic | 10,000 | 2,693 | 2,768 | 1.03 [1.04] |
| pairwise, exact | 15 | 3.16 | 11.2 | 3.53 [3.61] |
| pairwise, exact | 30 | 6.18 | 68.0 | 11.00 [11.73] |
| pairwise, exact | 49 | 8.21 | 277 | 33.77 [32.81] |

## References

- Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods".
  *Biometrics Bulletin*, 1(6), 80-83.
- Mann, H. B. & Whitney, D. R. (1947). "On a Test of Whether one of Two
  Random Variables is Stochastically Larger than the Other".
  *Annals of Mathematical Statistics*, 18(1), 50-60.
- Hodges, J. L. & Lehmann, E. L. (1963). "Estimates of Location Based on
  Rank Tests". *Annals of Mathematical Statistics*, 34(2), 598-611.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
