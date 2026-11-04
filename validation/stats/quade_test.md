# Validation: Quade Test (quade.test)

## Method Overview

The Quade test is a non-parametric test for unreplicated complete block designs, similar to the Friedman test. It uses a weighted ranking approach where blocks are weighted by their range, making it more powerful when block effects vary considerably.

**Key Features:**
- Uses F distribution (unlike Friedman's chi-squared approximation)
- Weights blocks by their range to recover between-block information
- More powerful than Friedman test when block ranges vary substantially
- Requires complete blocks (one observation per treatment per block)

**Mathematical Background:**
1. Compute the range within each block: Range_i = max(X_i) - min(X_i)
2. Rank the ranges across blocks: Q_i
3. Rank values within each block: R_ij
4. Compute weighted scores: S_ij = Q_i × (R_ij - (k+1)/2)
5. Compute treatment totals: S_j = Σ_i S_ij
6. F statistic: T = (b-1)B / (A - B)
   where A = ΣΣ S_ij² and B = (1/b) Σ S_j²

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | quade.test | R 4.3+ |

## Test Cases

Registry: `validation/methods/stats_rank_t3.json`, id `quade_ties_store` (R 4.3.3, cases of
`validation/r/cases/stats_rank_t3.R`). Tolerances: the `validation/README.md` rows for n < 100
(F, A, B and treatment sums `Tol::stat`, p `Tol::p_value`); df and block ranges exact.

### Test 1: Store-brand data (Conover 1999, p. 375; the `?quade.test` example)

Case `stat_rank_quade_store` (dataset `validation/datasets/generated/rank_quade_store.csv`).

```r
y <- matrix(c( 5,  4,  7, 10, 12,
               1,  3,  1,  0,  2,
              16, 12, 22, 22, 35,
               5,  4,  3,  5,  4,
              10,  9,  7, 13, 10,
              19, 18, 28, 37, 58,
              10,  7,  6,  8,  7),
            nrow = 7, byrow = TRUE,
            dimnames = list(Store = as.character(1:7), Brand = LETTERS[1:5]))
quade.test(y)
```

| Quantity | R value |
|---|---|
| Quade F | 3.8292515841753727 |
| df | 4, 24 |
| p-value | 0.015189020073274592 |
| treatment sums A..E | -9.5, -38, -14, 23.5, 38 |
| A | 1366.5 |
| B | 532.3571428571429 |
| block ranges | 8, 3, 23, 2, 6, 40, 4 |

A, B and the treatment sums are written out in the case as `quade.test.default` computes them.
The previous version of this test used a modified row 6 (19, 18, 28, 25, 20), for which R gives
F 2.4266430626244748 and p 0.075661171848975223, and quoted it as R's example.

**Rust test:** `stats::quade::tests::test_validate_quade_store`

### Test 2: Perfect treatment ordering

```r
y <- matrix(c(1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 1.8, 2.8, 3.8, 1.1, 2.1, 3.1),
            nrow = 5, byrow = TRUE)
quade.test(y)   # F = 36 on (2, 8), p = 1.0000000000000011e-04
```

Every block range is 2 in exact arithmetic; in floating point one differs (2.0 vs
1.9999999999999998), so the ranks of the ranges are not all tied, in R and in p2a alike. Property
test (df, F near 36, p < 0.05): `stats::quade::tests::test_quade_basic`.

### Test 3: Ties at any scale

Case `stat_rank_small_blocks` (8 blocks x 4 treatments of order 1e-12 with ties within blocks):
F 6.799802761341223 on (3, 21), p 0.002226306015893335, treatment sums t1..t4 -37, -23, 19, 41,
A 999.5, B 492.5.

**Rust test:** `stats::quade::tests::test_validate_quade_small_scale`

### Input checks

A NaN or infinite cell in `quade_test` is an error naming the block ("block 1 contains a missing
or infinite value"); `run_quade_test` drops incomplete blocks first. Test:
`stats::quade::tests::test_quade_rejects_non_finite`.

## Missing values, design checks and the R case

`run_quade_test(dataset, value, treatment, block)` pivots the long-format data through
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
quade.test(d$y, d$treat, d$block)
keep <- d$block != "S2"; quade.test(d$y_na[keep], d$treat[keep], d$block[keep])
```

| Quantity | R value |
|---|---|
| Quade F / df1 / df2 / p | 14.555694461807725 / 2 / 14 / 0.00038084840951921285 |
| with `y_na`, on the 7 complete blocks: F / df1 / df2 / p | 11.75532786885246 / 2 / 12 / 0.0014891392959195616 |
| R's NA path, `quade.test(d$y_na, ...)` (not pinned, see below): F / df2 / p | 9.639493014547027 / 14 / 0.0023318770707155983 |

**Rust tests** (`crates/p2a-core/src/stats/quade.rs`):
- `stats::quade::tests::test_validate_friedman_quade_quade` (F `Tol::stat(24)`, df exact, p
  `Tol::p_value(24)`; with `y_na` the complete-block values and 7 blocks)
- `stats::quade::tests::test_validate_quade_refuses_designs_r_refuses` (duplicate cell, missing
  block label, absent cell)
- `stats::quade::tests::test_validate_na_quade_dtypes_and_errors` (Int64 equals Float64; text
  and absent columns)
- MCP: `test_validate_mcp_hypothesis_quade_na` (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`)

## Numerical Precision Summary

- F, A, B and treatment sums are pinned at the `validation/README.md` rows (`Tol::stat`), the
  p-value at `Tol::p_value`; df and block ranges are exact.
- Ranks (within blocks and of the block ranges) come from `stats::ranks::midranks` (R's `rank()`,
  exact ties); the p-value is an upper tail computed directly (`traits::estimator::f_test_p_value`).

## Known Differences

1. **Tie handling in range ranking**: R uses exact equality for detecting ties in rankings. Floating-point differences (e.g., 2.0 vs 1.9999999999999998) result in different ranks. Rust matches this behavior.

2. **F distribution p-value**: Both use the F distribution CDF. Very small p-values may have slightly different precision.

3. **Missing responses (R 4.3.3 `quade.test`):** R removes a block with a missing response, as its
   help page says, but computes `b <- nlevels(blocks)` before that and never recomputes it, so the
   dropped block still counts in B, in `(b - 1)` and in the denominator df. On the case data R
   reports F 9.639493014547027 on (2, 14) df. p2a computes Quade on the complete blocks, b = 7:
   F 11.75532786885246 on (2, 12) df, p 0.0014891392959195616, which equals `quade.test` run on
   those 7 blocks, and that is the value pinned. R's NA-path values stay in the case JSON
   (`na_quade_f`, `na_quade_df2`, `na_quade_p`) unpinned. Friedman is unaffected (`friedman.test`
   recomputes `n <- nrow(y)` after dropping).

## Performance Comparison

| Configuration | Rust (µs) | R (µs) | Speedup |
|---------------|-----------|--------|---------|
| 10 blocks × 3 treatments | 1.89 | 670 | ~355x |
| 50 blocks × 4 treatments | 13.7 | 1,860 | ~136x |
| 100 blocks × 5 treatments | 27.0 | 3,610 | ~134x |
| 500 blocks × 3 treatments | 122 | 14,120 | ~116x |

**Notes:**
- Rust consistently outperforms R by 116-355x depending on problem size
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)
- Speedup is highest for small problems where R overhead dominates
- Performance measured on the same machine, same data generation seed

## References

- Quade, D. (1979). "Using weighted rankings in the analysis of complete blocks with additive block effects". Journal of the American Statistical Association, 74(367), 680-683.
- Conover, W. J. (1999). Practical Nonparametric Statistics (3rd ed.). New York: Wiley. Pages 373-380.
- R Core Team. `stats::quade.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quade.test.html
