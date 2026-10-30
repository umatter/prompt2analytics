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

### Test 1: RoundingTimes from R Documentation
**Description:** Classic example from R documentation with 12 blocks and 3 treatments.

**R Code:**
```r
RoundingTimes <- matrix(c(5.40, 5.50, 5.55,
                          5.85, 5.70, 5.75,
                          5.20, 5.60, 5.50,
                          5.55, 5.50, 5.40,
                          5.90, 5.85, 5.70,
                          5.45, 5.55, 5.60,
                          5.40, 5.40, 5.35,
                          5.45, 5.50, 5.35,
                          5.25, 5.15, 5.00,
                          5.85, 5.80, 5.70,
                          5.25, 5.20, 5.10,
                          5.65, 5.55, 5.45),
                        nrow=12, byrow=TRUE)
friedman.test(RoundingTimes)

# Expected output:
# Friedman chi-squared = 4.9787, df = 2, p-value = 0.08296
```

**Rust Test:** `crates/p2a-core/src/stats/friedman.rs::tests::test_validate_friedman_against_r`

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| Q statistic | 4.9787 | 4.9787 | 0.05 | PASS |
| df | 2 | 2 | exact | PASS |
| p-value | 0.08296 | 0.08296 | 0.01 | PASS |
| has_ties | TRUE | TRUE | - | PASS |

### Test 2: Manual Calculation Verification
**Description:** Simple 3x3 matrix for manual verification.

**Data:**
- 3 blocks, 3 treatments
- Block 1: [1, 2, 3] -> ranks [1, 2, 3]
- Block 2: [1, 2, 3] -> ranks [1, 2, 3]
- Block 3: [1, 2, 3] -> ranks [1, 2, 3]
- Rank sums: R1=3, R2=6, R3=9
- Q = (12/(3×3×4)) × (9+36+81) - 3×3×4 = 6

**Rust Test:** `crates/p2a-core/src/stats/friedman.rs::tests::test_validate_friedman_manual_calculation`

**Results Comparison:**

| Metric | Expected | Rust Value | Tolerance | Status |
|--------|----------|------------|-----------|--------|
| Q statistic | 6.0 | 6.0 | 0.01 | PASS |
| rank_sums | [3, 6, 9] | [3, 6, 9] | exact | PASS |
| has_ties | FALSE | FALSE | - | PASS |

### Test 3: Data with Ties
**Description:** Verifies tie correction is applied correctly.

**Rust Test:** `crates/p2a-core/src/stats/friedman.rs::tests::test_friedman_ties_correction_value`

**Results Comparison:**

| Metric | Expected | Rust Value | Status |
|--------|----------|------------|--------|
| has_ties | TRUE | TRUE | PASS |
| tie_correction | 0.75 | 0.75 | PASS |

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

- Q statistics match R within 0.01
- Degrees of freedom match exactly
- P-values match R's chi-squared approximation within 0.001
- Tie correction matches R's implementation

## Known Differences

1. **Chi-squared approximation:** Both R and our implementation use the chi-squared approximation, which is accurate when the number of blocks and treatments is not too small.

2. **Tie handling:** Both implementations apply tie correction using average (mid) ranks and adjust the variance.

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
