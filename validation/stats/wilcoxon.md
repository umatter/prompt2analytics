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
p2a's rank-sum `statistic` is the rank sum R1; R's W = R1 − n₁(n₁ + 1)/2 is compared through that
identity (STAT-72).

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

R: V = 27 (pinned), p = 0.6352893188352069; p2a's p is 0.6308000140667467 (its tie variance
uses the count of tied observations): known deviation STAT-42, not compared.

**Rust test (Tests 1-4):** `stats::wilcoxon::tests::test_validate_wilcoxon_examples`

### Test 5: N = 30 and n = 25: normal approximation, labelled

Rank sum on 15 + 15 tie-free values (set n30) and signed rank on 25 tie-free values without zeros
(set sr25), with the default configuration. p2a's default asks for the exact test below N = 50,
but its enumeration stops at N = 20, so it uses the normal approximation with continuity
correction and reports `exact = false` (the label 5f86b49 fixed; reverting it fails the test).

| Set | R (exact = FALSE, correct = TRUE) | pinned | R default (exact) |
|---|---|---|---|
| n30 | W 51 (rank sum 171), p 0.011400982911555176 | rank sum, W, p, `exact == false`, continuity flag | 0.009874822650594207 (STAT-73) |
| sr25 | V 231, p 0.06729929952386755 | V, p, `exact == false`, continuity flag | 0.06670159101486205 (STAT-73) |

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

**STAT-72 (registry deviation):** p2a's rank-sum `statistic` is the rank sum R1 of `x`; R reports
W = R1 - n_x (n_x + 1) / 2. The test pins R1 and compares `statistic - n (n + 1) / 2` with R's W
until STAT-72 (Tier 4) changes the reported statistic.

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

- Statistics (rank sum, W through the identity, V) are exact; p-values are pinned at the
  `validation/README.md` rows.
- Ranks come from `stats::ranks::midranks` (R's `rank()`, exact ties); normal tails are computed
  directly (no `1 - cdf`) and CI quantiles from `traits::estimator::normal_quantile`.
- The Hodges-Lehmann estimate and the confidence interval are not R-pinned (STAT-74).

## Known Differences

1. **Statistic (STAT-72, registry deviation):** p2a reports the rank sum R1 as `statistic`; R's W
   is R1 − n₁(n₁ + 1)/2 and is compared through that identity. `u_statistic` is not compared.

2. **Exact enumeration (STAT-73, registry deviation):** p2a enumerates the exact distribution
   only for N ≤ 20; R uses the exact test for N < 50 without ties. Between, p2a uses the normal
   approximation and says so (`exact = false`, Test 5).

3. **Tie variance (STAT-42, registry deviation):** p2a's tie correction of the variance uses the
   number of tied observations rather than Σ(t³ − t); statistics agree, p-values with ties
   differ (Test 4).

4. **Not yet pinned:** the sign of the `mu` shift in the rank-sum test (STAT-41) and the
   confidence interval, an approximation rather than R's inversion (STAT-74).

5. **Continuity correction:** both apply the ±0.5 correction by default in the normal
   approximation.

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
**Note: n=10 shows higher exact cost due to DP enumeration; larger n falls back to approximation.

## References

- Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods".
  *Biometrics Bulletin*, 1(6), 80-83.
- Mann, H. B. & Whitney, D. R. (1947). "On a Test of Whether one of Two
  Random Variables is Stochastically Larger than the Other".
  *Annals of Mathematical Statistics*, 18(1), 50-60.
- Hodges, J. L. & Lehmann, E. L. (1963). "Estimates of Location Based on
  Rank Tests". *Annals of Mathematical Statistics*, 34(2), 598-611.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html
