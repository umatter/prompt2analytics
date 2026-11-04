# Validation: Friedman Rank Sum Test

## Method Overview

The Friedman test is a non-parametric test for comparing more than two related groups (unreplicated blocked data). It is the non-parametric alternative to one-way repeated measures ANOVA.

**Key Parameters:**
- `data`: Matrix where rows are blocks and columns are treatments
- Degrees of freedom: k - 1 (where k = number of treatments)

**Test Statistic (Q):**
```
Q = (12 / (n*k*(k+1))) × Σ(Rj²) - 3*n*(k+1)
```

Where:
- n = number of blocks (subjects/rows)
- k = number of treatments (columns)
- Rj = sum of ranks for treatment j

**Tie Correction:**
```
Q_corrected = Q / (1 - Σ(ti³ - ti) / (n*k*(k² - 1)))
```

Where ti is the number of tied observations in the i-th tie group within each block.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | friedman.test | R 4.3+ |
| scipy.stats | Python | friedmanchisquare | 1.11+ |

## Test Cases

Registry: `validation/methods/stats_rank_t3.json`, id `friedman_ties_rounding` (R 4.3.3, cases of
`validation/r/cases/stats_rank_t3.R`). Tolerances: the `validation/README.md` rows for n < 100
(statistics `Tol::stat`, p `Tol::p_value`); df and rank sums exact.

### Test 1: RoundingTimes (Hollander & Wolfe 1973, p. 140; `?friedman.test`)

Case `stat_rank_friedman_rounding` (dataset `validation/datasets/generated/rank_friedman_rounding.csv`,
22 players x 3 methods; column `first12` marks the first 12 players).

```r
friedman.test(RoundingTimes)            # all 22 players
friedman.test(RoundingTimes[1:12, ])    # the first 12
```

| Quantity | 22 players | first 12 players |
|---|---|---|
| Friedman chi-squared | 11.142857142857142 | 4.9787234042553195 |
| df | 2 | 2 |
| p-value | 0.003805040775511363 | 0.08296290472378988 |
| rank sums (Round Out, Narrow Angle, Wide Angle) | 53, 47, 32 | 28.5, 25.5, 18 |

The previous version of this test showed the 12-player data under R's 22-player example and
compared it at tolerance 0.05.

**Rust test:** `stats::friedman::tests::test_validate_friedman_rounding`

### Test 2: Hand computation

3 blocks of (1, 2, 3): rank sums 3, 6, 9 and Q = (12 / 36) × 126 − 36 = 6 (asserted to 1e-12, rank
sums exactly).

**Rust test:** `stats::friedman::tests::test_friedman_statistic_hand_computed`

### Test 3: Tie correction by hand

3 blocks of (1, 1, 2): Σ(t³ − t) = 18, C = 1 − 18 / (3 · 3 · 8) = 0.75 (asserted to 1e-15).

**Rust test:** `stats::friedman::tests::test_friedman_ties_correction_value`

### Test 4: Ties at any scale (STAT-53)

Case `stat_rank_small_blocks` (8 blocks x 4 treatments, values of order 1e-12 with ties within
blocks; every block's range is below 1e-10). Before, values within 1e-10 were treated as tied, so
every block was one tie group and the test returned Q = 0, p = 1.

| Quantity | R value | p2a before |
|---|---|---|
| Friedman chi-squared (y and y_unit = y × 1e12) | 13.253164556962025 | 0 |
| df | 3 | 3 |
| p-value | 0.004119913441309317 | 1 |
| rank sums t1..t4 | 11.5, 16.5, 23, 29 | 20, 20, 20, 20 |

**Rust test:** `stats::friedman::tests::test_validate_friedman_small_scale`

### Test 5: Degenerate designs

- Every block tied, `friedman.test(matrix(c(1, 1, 1, 2, 2, 2), 2, byrow = TRUE))`: R gives NaN;
  p2a refuses the design (`InvalidSpecification`: "every block is tied ... the statistic is
  undefined"). Known deviation STAT-53 (the registry records the error in place of R's NaN).
- One tied block among untied ones, `friedman.test(matrix(c(1, 1, 1, 1, 2, 3, 2, 1, 3), 3, byrow =
  TRUE))`: 3, p 0.22313016014842982 (= exp(−1.5)) in R and p2a.
- NaN in a block of `friedman_test` is an error naming the block; `run_friedman_test` drops
  incomplete blocks first, as `friedman.test` does.

**Rust test:** `stats::friedman::tests::test_friedman_all_tied_blocks_and_nan`

## Missing values, design checks and the R case

`run_friedman_test(dataset, value, treatment, block)` pivots the long-format data through
`unreplicated_blocks` (`crates/p2a-core/src/stats/friedman.rs`, shared by Friedman and Quade),
which applies R's rules in R's order:

1. The value column must be numeric: integers are accepted, a text column is
   `NonNumericColumn`, an absent column `ColumnNotFound`.
2. A missing treatment or block label is an error, "NA's are not allowed in 'groups' or
   'blocks'" (the treatment column is checked first, then the block column), whatever the
   response of that row is.
3. A (block, treatment) cell with more than one observation, or with none, is "not an
   unreplicated complete block design" (the message names the block and the treatment).
4. A block with a missing response is dropped, as R's `complete.cases` on the block x
   treatment table, and reported in `missing` (all rows of the block count as dropped;
   `by_column` counts the missing response cells).

Before T2-WP07 a replicated cell kept its last value without notice, a block without some
treatment was skipped without notice, a missing label became a block or treatment named
`null`, text labels kept their quotes and an Int64 response was refused.

**R case** `stat_friedman_quade` (`validation/r/cases/stats_anova.R`, dataset
`validation/datasets/generated/friedman_quade.csv`): 8 blocks x 3 treatments with an integer
response `y` (Int64 in polars), and `y_na` missing in row 5 (block S2). The case also checks
with `stopifnot` that R refuses a duplicate (S1, A) row, a missing block label and an absent
cell, in both tests.

```r
friedman.test(d$y, d$treat, d$block); friedman.test(d$y_na, d$treat, d$block)
```

| Quantity | R value |
|---|---|
| Friedman chi-squared / df / p | 13.03448275862069 / 2 / 0.0014777400088913251 |
| with `y_na` (block S2 dropped): chi-squared / p | 11.12 / 0.003848776397610543 |
| complete blocks with `y_na` | 7 |

**Rust tests** (`crates/p2a-core/src/stats/friedman.rs`):
- `stats::friedman::tests::test_validate_friedman_quade_friedman` (statistic `Tol::stat(24)`, df
  exact, p `Tol::p_value(24)`, 7 blocks with `y_na`, 3 of 24 rows dropped)
- `stats::friedman::tests::test_validate_friedman_refuses_designs_r_refuses` (duplicate cell,
  missing block label, missing treatment label, absent cell, missing label and response in one row)
- `stats::friedman::tests::test_validate_na_friedman_dtypes_and_errors` (Int64 equals Float64;
  text and absent columns)
- MCP: `test_validate_mcp_hypothesis_friedman_na` (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`;
  includes the refusal of a replicated cell)

## Numerical Precision Summary

- Statistics and p-values are pinned at the `validation/README.md` rows; df and rank sums are
  exact.
- Blocks are ranked with `stats::ranks::midranks` (R's `rank()`, exact ties); p-values are upper
  tails computed directly (`traits::estimator::chi_squared_p_value`).

## Known Differences

1. **Chi-squared approximation:** Both R and our implementation use the chi-squared approximation, which is accurate when the number of blocks and treatments is not too small.

2. **Tie handling:** Both implementations apply tie correction using average (mid) ranks and adjust the variance.

3. **Every block tied (STAT-53, registry deviation):** p2a returns an error where
   `friedman.test` returns NaN.

## Performance Comparison

Benchmarked on 2026-01-20.

| Blocks | Rust (µs) | R (µs)  | Speedup |
|--------|-----------|---------|---------|
| n=30   | 6.7       | 3,820   | ~570x   |
| n=100  | 18.0      | 15,500  | ~860x   |
| n=300  | 47.3      | 45,120  | ~954x   |
| n=1000 | 192.7     | 159,640 | ~828x   |

**Performance Evaluation:**
- All sizes show Rust significantly faster than R: PASS
- At least 2 sizes >= 2x speedup: PASS (all 4 sizes show 570-950x)
- n=300 (typical use case) >= 1.5x speedup: PASS (~954x)

## References

- Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance". Journal of the American Statistical Association, 32(200), 675-701.
- Hollander, M. & Wolfe, D. A. (1973). *Nonparametric Statistical Methods*. New York: John Wiley & Sons. Pages 139-146.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/friedman.test.html
