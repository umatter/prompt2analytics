# Validation: Multivariate Analysis of Variance (MANOVA)

## Method Overview

MANOVA (Multivariate Analysis of Variance) extends univariate ANOVA to multiple response variables. Instead of comparing scalar group means, we compare mean vectors, accounting for correlations between response variables.

**Key Parameters:**
- `y_data`: Matrix of response variables (n × p)
- `groups`: Group assignments for each observation
- `test`: Test statistic type (Pillai, Wilks, Hotelling-Lawley, Roy)

**Test Statistics (based on eigenvalues λᵢ of E⁻¹H):**
- **Wilks' Lambda:** Λ = ∏(1/(1+λᵢ)) - most popular
- **Pillai's Trace:** V = Σ(λᵢ/(1+λᵢ)) - most robust
- **Hotelling-Lawley Trace:** T² = Σλᵢ
- **Roy's Largest Root:** θ = λ₁

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `manova()`, `summary.manova()` | R 4.3.2 |

## Test Cases

### Test 1: Three Groups with Clear Separation

Three groups with clearly separated mean vectors on two response variables.

**R Code:**
```r
# Three groups with clear separation
y1 <- c(1.0, 1.2, 0.8, 5.0, 5.2, 4.8, 9.0, 9.2, 8.8)
y2 <- c(8.0, 7.8, 8.2, 4.0, 4.2, 3.8, 1.0, 1.2, 0.8)
group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))

fit <- manova(cbind(y1, y2) ~ group)
summary(fit, test = "Wilks")
summary(fit, test = "Pillai")
summary(fit, test = "Hotelling-Lawley")
summary(fit, test = "Roy")
```

**Expected Results:**
- All four tests should be highly significant (p < 0.001)
- Wilks' Lambda should be close to 0 (strong group differences)
- Pillai's Trace should be close to 2 (maximum with 3 groups, 2 variables)

**Results Comparison:**

| Metric | R | Rust (p2a) | Tolerance |
|--------|---|------------|-----------|
| Wilks' Lambda | < 0.01 | < 0.01 | - |
| Pillai's Trace | > 1.5 | > 1.5 | - |
| p-value (Pillai) | < 0.05 | < 0.05 | - |

**Rust Test:** `crates/p2a-core/src/stats/manova.rs::tests::test_manova_separated_groups_are_significant` (a property check: small Wilks' Lambda, significant Wilks and Pillai tests, non-negative eigenvalues; it compares no R value)

### Test 2: Two Groups - Basic MANOVA

Simple two-group comparison with two response variables.

**R Code:**
```r
y <- matrix(c(
  1.0, 2.0,
  1.2, 2.1,
  1.1, 1.9,
  3.0, 4.0,
  3.2, 4.1,
  2.9, 3.9
), ncol = 2, byrow = TRUE)
group <- factor(c("A", "A", "A", "B", "B", "B"))

fit <- manova(y ~ group)
summary(fit, test = "Pillai")
```

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| n_groups | 2 | 2 | exact |
| n_responses | 2 | 2 | exact |
| p-value | < 0.05 | < 0.05 | - |

**Rust Test:** `crates/p2a-core/src/stats/manova.rs::tests::test_manova_basic`

### Test 3: No Difference Between Groups

Two groups with overlapping distributions - should not reject H₀.

**R Code:**
```r
y <- matrix(c(
  1.0, 5.0,
  1.5, 4.5,
  2.0, 6.0,
  0.8, 5.2,
  1.2, 4.8,
  1.7, 5.5,
  1.3, 4.7,
  1.1, 5.3
), ncol = 2, byrow = TRUE)
group <- factor(c("A", "A", "A", "A", "B", "B", "B", "B"))

fit <- manova(y ~ group)
summary(fit, test = "Wilks")
```

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| Wilks' Lambda | > 0.5 | > 0.5 | - |
| p-value | > 0.05 | > 0.05 | - |

**Rust Test:** `crates/p2a-core/src/stats/manova.rs::tests::test_manova_no_difference`

### Test 4: Eigenvalue Computation

Direct verification of eigenvalue computation for E⁻¹H.

**Setup:**
```
H = [[4, 2], [2, 4]]
E = [[2, 0], [0, 2]]

E⁻¹H = [[2, 1], [1, 2]]
Eigenvalues should be 3 and 1
```

**Results Comparison:**

| Metric | Expected | Rust (p2a) | Tolerance |
|--------|----------|------------|-----------|
| λ₁ (max) | 3.0 | 3.0 | < 0.001 |
| λ₂ (min) | 1.0 | 1.0 | < 0.001 |

**Rust Test:** `crates/p2a-core/src/stats/manova.rs::tests::test_eigenvalue_computation`

### Test 5: Test Statistic Formulas

Direct verification of test statistic computations with known eigenvalues [3, 1].

**Expected Results:**
- Wilks' Lambda = (1/4) × (1/2) = 0.125
- Pillai's Trace = 0.75 + 0.5 = 1.25
- Hotelling-Lawley = 3 + 1 = 4
- Roy's Largest Root = 3

**Rust Tests:**
- `crates/p2a-core/src/stats/manova.rs::tests::test_wilks_lambda_formula`
- `crates/p2a-core/src/stats/manova.rs::tests::test_pillai_trace_formula`
- `crates/p2a-core/src/stats/manova.rs::tests::test_hotelling_lawley_formula`
- `crates/p2a-core/src/stats/manova.rs::tests::test_roy_largest_root_formula`

## Missing values and the R case

`run_manova(dataset, responses, group)` reads the responses and the group in one request through
the shared extractor (`p2a_core::data::columns`): a row with a missing response or group label is
dropped from all of them (joint deletion, as `manova`, an `lm`, does with `na.omit`) and reported
in `missing`. Integer responses are accepted, and group labels are R `factor()` labels (text as
stored, without quotes). Before T2-WP07 a missing response was written into the data matrix as
NaN, and the call failed with an internal error ("Failed to invert error SSCP matrix: Matrix
contains non-finite values (NaN or Inf)").

**R case** `stat_manova_na` (`validation/r/cases/stats_anova.R`, dataset
`validation/datasets/generated/manova_na.csv`, 36 rows in 3 groups; `y1` is missing in rows 3
and 20 and `y2` in row 30, so 33 rows are used):

```r
m <- manova(cbind(y1, y2) ~ g, data = d)
summary(m, test = "Pillai" | "Wilks" | "Hotelling-Lawley" | "Roy")$stats
```

| Statistic | value | approx F | df1 / df2 | p |
|---|---|---|---|---|
| Pillai | 0.8666827691016671 | 11.470964335572894 | 4 / 60 | 5.569040967348833e-07 |
| Wilks | 0.3033643446479518 | 11.826051982166383 | 4 / 58 | 4.319740428530747e-07 |
| Hotelling-Lawley | 1.7358287184788463 | 12.150801029351925 | 4 / 56 | 3.5354429854995737e-07 |
| Roy | 1.3069334489125344 | 19.604001733688015 | 2 / 30 | 3.584874936452156e-06 |

Rows used: 33; group sizes g1 11, g2 11, g3 11. The F approximations are those of
`summary.manova`, term by term.

**Rust tests** (`crates/p2a-core/src/stats/manova.rs`):
- `stats::manova::tests::test_validate_manova_na` (all four statistics, approximate F and p at
  n = 33, df exact except Wilks' df2 at `Tol::stat(33)` since Rao's df is real-valued in
  general; rows used and group sizes; `missing.by_column` y1: 2, y2: 1)
- `stats::manova::tests::test_validate_na_manova_dtypes_and_errors` (the Int64 column `id` as a
  response equals its Float64 cast; a text response is `NonNumericColumn`, an absent response
  `ColumnNotFound`)
- MCP: `test_validate_mcp_anova_manova_na` (`crates/p2a-mcp/tests/tier2/wp07_anova.rs`; Pillai
  and Wilks, rows used, group labels g1, g2, g3)

## Numerical Precision Summary

| Test Case | Statistic Match | P-value Match |
|-----------|-----------------|---------------|
| Three groups separated | Exact formulas | Approximate (F-approx differs) |
| Two groups basic | Comparable | Comparable |
| No difference | Comparable | Comparable |
| Eigenvalues | < 0.001 | N/A |
| Test formulas | < 0.001 | N/A |

## Known Differences

1. **F-Approximation Methods**: R uses specific approximation formulas that may differ slightly from our implementation. All four statistics converge for large samples.

2. **Exact vs Approximate**: F-tests are exact when s = min(p, v_h) ≤ 2, otherwise approximate.

3. **Eigenvalue Computation**: We use the Cholesky decomposition method (E = LL^T, then compute eigenvalues of L⁻¹HL⁻ᵀ) for numerical stability.

4. **Ties and Singularity**: When response variables are perfectly correlated within groups, the error matrix E becomes singular. Our implementation checks for this and returns an error.

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100, p=2, g=3 | 23 | 1,160 | ~50x |
| n=1,000, p=3, g=4 | 146 | 1,380 | ~9x |
| n=10,000, p=5, g=5 | 1,808 | 7,200 | ~4x |

*Benchmarks run on Linux with Rust Criterion (100 samples) and R system.time (50 iterations). Speedup decreases at larger n due to matrix operations dominating both implementations.*

## MCP Tool Usage

```json
{
  "tool": "anova_manova",
  "dataset": "my_data",
  "response_vars": ["score1", "score2", "score3"],
  "factor": "treatment",
  "test": "pillai"
}
```

## References

- Wilks, S. S. (1932). "Certain generalizations in the analysis of variance". *Biometrika*, 24(3-4), 471-494.
- Pillai, K. C. S. (1955). "Some new test criteria in multivariate analysis". *Annals of Mathematical Statistics*, 26, 117-121.
- Lawley, D. N. (1938). "A generalization of Fisher's z test". *Biometrika*, 30, 180-187.
- Roy, S. N. (1939). "p-statistics and some generalizations in analysis of variance". *Sankhya*, 4, 381-396.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/manova.html
- R-bloggers: https://www.r-bloggers.com/2016/12/manova-test-statistics-with-r/
