# Validation: Mahalanobis Distance

## Method Overview

The `mahalanobis()` function computes the squared Mahalanobis distance of observations from a center point, accounting for the covariance structure of the data.

**Key Features:**
- Measures multivariate distance accounting for correlations
- Useful for outlier detection
- Under multivariate normality, D² ~ χ²(p)
- Supports pre-computed inverse covariance for efficiency

**Mathematical Background:**

The squared Mahalanobis distance is defined as:

```
D² = (x - μ)' Σ⁻¹ (x - μ)
```

where:
- x is the observation vector
- μ is the center (typically the mean)
- Σ is the covariance matrix

When Σ = I (identity), Mahalanobis distance reduces to Euclidean distance.

**Implementation:**

The Rust implementation uses Cholesky decomposition for efficient computation:
1. Decompose Σ = LL' (Cholesky)
2. Solve Lz = (x - μ) via forward substitution
3. D² = z'z = ||z||²

This avoids explicit matrix inverse and is more numerically stable. A rank-deficient covariance is an error by default (see "Singular covariance").

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | mahalanobis | R 4.3+ |

## Test Cases

### Test 1: Identity Covariance (Euclidean Distance)

**R Code:**
```r
x <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
center <- c(0.5, 0.5)
cov_mat <- diag(2)
mahalanobis(x, center, cov_mat)
# Expected: [0.5, 0.5, 0.5, 0.5] (all equidistant from center)
```

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_basic`

### Test 2: With Correlation

**Test:** Verify distances with correlated variables differ from uncorrelated case.

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_with_correlation`

### Test 3: Auto-compute Center and Covariance

**Test:** Let function compute center and covariance from data.

**R Code:**
```r
x <- matrix(c(1, 2, 3, 5, 2, 4, 5, 3), ncol = 2, byrow = TRUE)
center <- colMeans(x)  # [2.75, 3.5]
cov_mat <- cov(x)
mahalanobis(x, center, cov_mat)
```

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_distance_sum_identity`

### Test 4: Pre-inverted Covariance

**Test:** Verify same results when providing inverse covariance directly.

**Rust Test:** `crates/p2a-core/src/stats/mahalanobis.rs::tests::test_mahalanobis_inverted`

## Singular covariance (R-pinned)

R's `mahalanobis` stops on a rank-deficient covariance ("system is exactly singular"), and so do
`mahalanobis` and `run_mahalanobis` (`SingularMatrix`, naming the rank). A silent pseudoinverse would
report distances whose chi-square df is the rank, not p, under a threshold written for p.
`mahalanobis_with` and `run_mahalanobis_with` take `SingularCovariance::Generalized`: the distances use
a generalized inverse, `rank` holds the numerical rank and `warnings` says so. On a sample covariance
these are `MASS::ginv`'s distances, because the centred rows lie in the covariance's range.
`MahalanobisResult::outlier_threshold(level)` is `qchisq(level, rank)`. A given `cov` must be symmetric
(R's `solve` does not check); an estimated covariance needs at least two rows.

**R Code:** case `stat_mahalanobis_t3` in `validation/r/cases/stats_misc_t3.R`, dataset
`validation/datasets/generated/mahalanobis_t3.csv`.

| Quantity | R |
|---|---|
| full rank, centre (`f1`, `f2`, `f3`) | 10.009863333333334, 4.71697, -0.9299966666666667 |
| full rank, `qchisq(0.975, 3)` | 9.348403604496145 |
| b = 2 a (rank 1), first `ginv` distances | 1.445901639344263, 0.39672131147541, 0.0032786885245901704 |
| b = 2 a, `qchisq(0.975, 1)` against `qchisq(0.975, 2)` | 5.023886187314885, 7.377758908227871 |
| total = sum of parts (rank 2 of 3), first `ginv` distances | 0.8617761007344652, 0.7505022370089935, 0.5086063079427657 |
| total, `qchisq(0.975, 2)` against `qchisq(0.975, 3)` | 7.377758908227871, 9.348403604496145 |

**Rust tests:**
- `stats::mahalanobis::tests::test_validate_mahalanobis_t3_full_rank` (every distance, the centre, the rank and threshold)
- `stats::mahalanobis::tests::test_validate_mahalanobis_t3_singular_is_error` (both singular designs refuse; R's `solve` stops on them)
- `stats::mahalanobis::tests::test_validate_mahalanobis_t3_generalized` (`ginv` distances, rank, threshold, warning)
- `stats::mahalanobis::tests::test_mahalanobis_needs_two_rows_for_covariance`

**Status:** R-pinned (registry `validation/methods/stats_misc_t3.json`, id `mahalanobis_t3`).

The MCP tool's threshold is a separate item (TOOL-10 below); `outlier_threshold` is what it can call.

## Missing values (R-pinned)

`run_mahalanobis(dataset, columns, center, cov)` reads the columns through the shared extractor
in one request and drops every row with a missing value (null or NaN) in any of them. The centre
and the covariance are computed from the kept rows, and the distances are reported for the kept
rows only. `MahalanobisResult.rows` gives the original (0-based) row of each distance and
`MahalanobisResult.missing` what was dropped; the array-level `mahalanobis()` sets `rows` to
0..n. Integer columns are accepted; a text column is `NonNumericColumn`. Before T2-WP06 a null
was read as 0, which moved the centre, inflated the covariance and produced a spurious outlier
(a distance of 14.757 for row 6 of the case below).

**R Code** (case `stat_pairwise_mahalanobis_na` in `validation/r/cases/stats_onesample.R`;
dataset `validation/datasets/generated/pairwise_mahalanobis_na.csv`, 30 rows, `a` missing in
row 7, 1-based):
```r
set.seed(9)   # columns as in the case file
cc <- complete.cases(d$a, d$b); X <- as.matrix(d[cc, c("a", "b")])
md <- mahalanobis(X, colMeans(X), cov(X))
```

| Quantity | R |
|---|---|
| distances | 29 values, the first five 0.7548069130399198, 1.5348361624055533, 1.5870751414280628, 0.22951689971894934, 0.3896433925385543 (all in `validation/expected/stat_pairwise_mahalanobis_na.json`) |
| kept rows (0-based) | 0-29 without 6 |
| centre a, b | 4.871103448275862, 2.411896551724138 |
| n | 29 |

**Rust tests:**
- `stats::mahalanobis::tests::test_validate_mahalanobis_na` (every distance, the rows, the centre, n)
- `stats::mahalanobis::tests::test_validate_na_mahalanobis_columns` (Int64 against Float64, text and absent columns)
- `tier2::wp06_onesample::test_validate_mcp_multivariate_mahalanobis` (p2a-mcp: `n_obs`, `center`, the ten printed `first_10_distances` and `summary.mean_distance` at `Tol::printed(4)`)

**MCP tool: row labels and missing values.** `multivariate_mahalanobis` numbers each of its
`first_10_distances` by its 1-based dataset row (`MahalanobisResult.rows` + 1), so on this case
the ten observations are 1-6 and 8-11 (row 7 has a missing `a`); before, they were numbered
1-10 by kept row. The tool reports the drop under `missing` (`{"n_total": 30, "n_used": 29,
"n_dropped": 1, "by_column": {"a": 1}}`).
`tools::handlers::stats::tests::test_validate_mcp_multivariate_mahalanobis_rows` checks the row
numbers against `which(complete.cases(a, b))`, the ten printed distances against R at
`Tol::printed(4)` and the rows used exactly (registry `validation/methods/mcp_stats.json`, id
`mcp_multivariate_mahalanobis_rows`).

**Known difference (TOOL-10, Tier 4):** the tool's `chi_squared_threshold_95` and
`n_potential_outliers` use the threshold 2.7 p, not `qchisq(0.95, p)`; they are not pinned.

**Tolerance:** `validation/README.md`, n < 100: distances 1e-5 relative (`Tol::stat`), centre
1e-6 (`Tol::coef`); rows and counts exact.

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`, id `mahalanobis`).

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| Squared distances | 1e-10 | High precision |
| Center computation | 1e-10 | Mean computation |
| Covariance | 1e-10 | Sample covariance |

## Known Differences

1. **Covariance formula**: Both use n-1 denominator (sample covariance).

2. **Matrix inverse**: R uses `solve()`, Rust uses LU decomposition via faer.

3. **Singular matrices**: Both refuse a rank-deficient covariance. `SingularCovariance::Generalized` is p2a's explicit alternative.

## Performance Comparison

### Sample Size Scaling (p=5)

| n | Rust (µs) | R (µs) | Speedup |
|---|-----------|--------|---------|
| 100 | 23 | 150 | 6.5x |
| 1,000 | 215 | 250 | 1.2x |
| 10,000 | 2,149 | 1,300 | 0.6x |
| 100,000 | 21,451 | 11,810 | 0.55x |

### Variable Scaling (n=1,000)

| p | Rust (µs) | R (µs) | Speedup |
|---|-----------|--------|---------|
| 2 | 180 | 100 | 0.56x |
| 5 | 210 | 220 | 1.05x |
| 10 | 321 | 410 | 1.28x |
| 20 | 568 | 940 | 1.65x |
| 50 | 1,590 | 3,900 | 2.45x |

**Notes:**
- Rust benchmarks from Criterion (median), R from microbenchmark (mean of 100 iterations)
- Rust uses Cholesky decomposition + forward substitution (optimized Jan 2026)
- R uses highly optimized BLAS routines via `solve()` for matrix operations
- **Rust excels for high-dimensional data (p >= 10)**: 1.3-2.5x faster
- **R excels for large n with small p**: 1.5-2x faster due to BLAS vectorization
- For typical use cases (n < 10,000, p >= 5), Rust performance is comparable or better
- Cholesky optimization improved high-p performance by ~90% (p=50: 15ms → 1.6ms)

## References

- Mahalanobis, P. C. (1936). "On the generalized distance in statistics".
  Proceedings of the National Institute of Sciences (Calcutta), 2, 49–55.
- R Core Team. `stats::mahalanobis()` function.
  https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mahalanobis.html
