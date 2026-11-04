# Validation: Factor Analysis (factanal)

## Method Overview

Maximum Likelihood Factor Analysis (MLFA) identifies latent factors underlying observed variables by fitting the model:

```
x = Λf + e
```

where:
- x is a p-element vector of observations
- Λ is a p×k matrix of factor loadings
- f is a k-element vector of factor scores (uncorrelated, unit variance)
- e is a p-element error vector with variances Ψ (uniquenesses)

The implied correlation structure is: `Σ = ΛΛ' + Ψ`

**Key Parameters:**
- `n_factors`: Number of latent factors to extract
- `rotation`: Rotation method (varimax, promax, none)
- `scores`: Factor score computation method (none, regression, bartlett)

## Rotations and sorting (R-pinned)

- **varimax** (`stats::factanal::varimax`): R's `stats::varimax(x, normalize = TRUE)` algorithm, the SVD
  iteration with Kaiser normalisation: `z = x T`, `B = x' (z^3 - z diag(colSums(z^2)) / p)`, `T = U V'` from
  the SVD of B, stopping when the sum of singular values grows by less than 1e-14 relative (at most 1000
  iterations). Varimax is R-pinned at the optimum (R's `varimax(eps = 1e-14)`). R's default output (and `factanal`'s)
  stops at eps = 1e-5, up to 7.1e-4 from the optimum in a loading on `stat_factanal_promax` (promax: 6.9e-5),
  so R's printed 3 decimals can differ in the last digit.
- **promax** (`promax(x, m)`, m = 4 as R's default): varimax, the target `Q = xv |xv|^(m - 1)`,
  `U = lm.fit(xv, Q)` (pivoted QR, `linalg::solve::least_squares`), `U = U diag(sqrt(diag(solve(U'U))))`;
  loadings `xv U`, rotation matrix `T_v U`.
- **Factor correlations** (`factor_correlation(T)`): `solve(T) solve(T)' = (T'T)^-1`.
- **`sortLoadings`** (`sort_loadings`): for every rotation, including none, the columns are ordered by
  descending sum of squared loadings (a stable sort) and a column with a negative sum is negated; the
  factor correlations get the same permutation P and signs D (`D P' Phi P D`), so they are in the
  loadings' order.
- **Communalities** are `rowSums(L0^2)` of the unrotated loadings, which equals
  `diag(Lambda Phi Lambda')` for every rotation (not the row sums of squared promax pattern loadings).
- The p-value is the chi-square upper tail (`chi_squared_p_value`).

### R case `stat_factanal_promax`

`validation/r/cases/stats_power_loglin.R`, dataset `validation/datasets/generated/factanal_promax.csv`
(n = 500, 12 variables, 3 factors). R: `factanal(X, 3, rotation = "none" | "varimax" | "promax",
control = list(opt = list(factr = 1, pgtol = 0)))`, and on R's own unrotated loadings
`varimax(L0, eps = 1e-14)` with promax's four lines on it, after `sortLoadings`' order and signs.

Tests (`p2a-core`, `stats::factanal::tests::`):

- `test_validate_factanal_rotation_promax`: the rotations of R's unrotated loadings (`rot_varimax_tight`,
  `rot_varimax_rotmat_tight`, `rot_promax_tight`, `rot_promax_rotmat_tight`, `rot_promax_phi_tight`) at
  `Tol::iterative(1e-8)`, and the normalised varimax criterion at least R's tight one minus 1e-12 (no
  other local optimum). The keys of R's default eps 1e-5 iterate (`rot_varimax`, `rot_promax`, ...) are
  not read: not R-pinned.
- `test_validate_factanal_against_r`: the full fit for the three rotations at `Tol::iterative(1e-4)`;
  p2a's fixed point at tolerance 1e-12 against R's L-BFGS-B. Unrotated: loadings and variance
  proportions against `factanal`'s; uniquenesses, communalities, chi-square, p, objective, dof (exact).
  Varimax and promax: loadings, variance proportions and promax factor correlations against R's
  rotations at the optimum (`rot_*_tight`), since `factanal`'s own rotated output is R's eps 1e-5
  iterate, up to 7.1e-4 away (registry deviation STAT-23).
- `test_rotation_and_scores_methods_parse_aliases_and_reject_unknown` (not R-pinned): option spellings
  (`varimax`, `promax`, `none`; scores `none`, `regression`/`thomson`, `bartlett`).

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `factanal()` | R 4.3.x |
| psych | R | `fa()` | 2.3.x |
| scikit-learn | Python | `FactorAnalysis` | 1.3.x |

## Test Cases

### Test 1: Synthetic Two-Factor Structure

**Data Generation:**
Six variables with known factor structure:
- Variables 1-3 load on Factor 1 (loadings ~0.7-0.9)
- Variables 4-6 load on Factor 2 (loadings ~0.7-0.9)

**R Code:**
```r
set.seed(42)
n <- 200
f1 <- rnorm(n)
f2 <- rnorm(n)

# Generate data with known factor structure
data <- data.frame(
  x1 = 0.8 * f1 + sqrt(1-0.8^2) * rnorm(n),
  x2 = 0.7 * f1 + sqrt(1-0.7^2) * rnorm(n),
  x3 = 0.75 * f1 + sqrt(1-0.75^2) * rnorm(n),
  x4 = 0.8 * f2 + sqrt(1-0.8^2) * rnorm(n),
  x5 = 0.7 * f2 + sqrt(1-0.7^2) * rnorm(n),
  x6 = 0.75 * f2 + sqrt(1-0.75^2) * rnorm(n)
)

# Run factor analysis
result <- factanal(data, factors = 2, rotation = "varimax")
print(result)
print(result$uniquenesses)
print(result$loadings)
```

**Results Comparison:**

| Metric | R (factanal) | Rust (p2a) | Tolerance |
|--------|--------------|------------|-----------|
| Uniquenesses | 0.1-0.5 range | 0.1-0.5 range | ±0.1 |
| Communalities | 0.5-0.9 range | 0.5-0.9 range | ±0.1 |
| Chi-squared | > 0 | > 0 | ±0.5 |
| Convergence | Yes | Yes | - |

**Rust Test:** `crates/p2a-core/src/stats/factanal.rs::tests::test_factanal_basic`

### Test 2: Varimax Rotation Properties

**Test Property:** Varimax rotation should preserve communalities while maximizing variance of squared loadings.

**R Code:**
```r
# Test that communalities are preserved after rotation
loadings_raw <- matrix(c(
  0.7, 0.3,
  0.6, 0.4,
  0.65, 0.35,
  0.3, 0.7,
  0.4, 0.6,
  0.35, 0.65
), nrow = 6, byrow = TRUE)

# Compute communalities before rotation
h2_before <- rowSums(loadings_raw^2)

# Apply varimax
rotated <- varimax(loadings_raw)$loadings

# Communalities after rotation
h2_after <- rowSums(rotated^2)

# Should be equal
all.equal(h2_before, h2_after, tolerance = 1e-6)
```

**Results Comparison:**

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Communality preservation | Equal before/after | 1e-4 |
| Rotation determinant | 1.0 (orthogonal) | 1e-6 |

**Rust Test:** `crates/p2a-core/src/stats/factanal.rs::tests::test_varimax_rotation`

### Test 3: Chi-Squared Goodness-of-Fit Test

**Test Property:** Chi-squared test for model adequacy.

**R Code:**
```r
set.seed(42)
n <- 100
data <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  x4 = rnorm(n),
  x5 = rnorm(n),
  x6 = rnorm(n)
)

# Add correlations
data$x2 <- 0.7*data$x1 + 0.3*rnorm(n)
data$x3 <- 0.7*data$x1 + 0.3*rnorm(n)
data$x5 <- 0.7*data$x4 + 0.3*rnorm(n)
data$x6 <- 0.7*data$x4 + 0.3*rnorm(n)

result <- factanal(data, factors = 2)

# Degrees of freedom: ((p-k)^2 - p - k) / 2 = ((6-2)^2 - 6 - 2) / 2 = 4
print(result$STATISTIC)  # Chi-squared
print(result$PVAL)       # p-value
print(result$dof)        # df = 4
```

**Results Comparison:**

| Metric | R | Rust | Tolerance |
|--------|---|------|-----------|
| df | 4 | 4 | 0 |
| Chi-squared | > 0 | > 0 | ±1.0 |
| p-value | (0, 1) | (0, 1) | ±0.1 |

**Rust Test:** `crates/p2a-core/src/stats/factanal.rs::tests::test_factanal_chi_squared`

## Numerical Precision Summary

| Component | Tolerance (`stat_factanal_promax`) |
|-----------|-------------------|
| Rotations of R's unrotated loadings (varimax, promax, rotation matrices, Phi) | `Tol::iterative(1e-8)` |
| Loadings, uniquenesses, communalities, variance proportions, Phi (full fit) | `Tol::iterative(1e-4)` |
| Chi-squared, p-value, objective | `Tol::iterative(1e-4)` |
| Degrees of freedom | exact |

The synthetic tests 1-3 above check properties, not R values.

## Known Differences

1. **Starting Values**: R uses Jöreskog (1963) initialization. Rust uses eigenvalue-based initialization. Both converge to similar solutions.

2. **Optimization Algorithm**: R uses `optim()` with L-BFGS-B. Rust uses EM-style alternating optimization. Results are equivalent within tolerance.

3. **Varimax stop**: p2a iterates varimax to its optimum (relative 1e-14); R's default stops at eps = 1e-5,
   up to 7.1e-4 away in a loading. The rotated full fits are therefore pinned to R's tight rotation keys,
   not to `factanal`'s default output.

4. **Promax power**: m = 4, R's default.

5. **Missing values**: `run_factanal` refuses a null or NaN cell with `NullValues` naming the column, as R's `factanal` refuses NA; integer columns are accepted (T2-WP06; `stats::factanal::tests::test_na_factanal_columns`). That test is not R-pinned; the rotations and the fit are (above).

6. **Uniqueness optimiser**: p2a's own fixed point, pinned at 1e-4 relative to R's L-BFGS-B optimum; R's
   default `factr` itself moves R's uniquenesses by about 7e-6.

## Performance Comparison

Benchmarks measured on 2026-01-20 (v2 with optimizations). Rust times from Criterion, R times from system.time().

### Varimax Rotation (Primary Comparison)

| Dataset (n, p, k) | Rust (ms) | R (ms) | Speedup |
|-------------------|-----------|--------|---------|
| n=100, p=6, k=2   | 0.49      | 8.64   | **~18x**  |
| n=500, p=10, k=3  | 1.03      | 4.70   | **~4.6x** |
| n=1000, p=15, k=4 | 2.74      | 7.44   | **~2.7x** |
| n=5000, p=20, k=5 | 7.19      | 7.80   | **~1.1x** |

### All Rotation Methods

| Dataset | Method | Rust (ms) | R (ms) |
|---------|--------|-----------|--------|
| n=100, p=6, k=2 | No rotation | 0.45 | 2.60 |
| n=100, p=6, k=2 | Varimax | 0.49 | 8.64 |
| n=100, p=6, k=2 | Promax | 0.45 | 11.54 |
| n=500, p=10, k=3 | No rotation | 1.02 | 3.56 |
| n=500, p=10, k=3 | Varimax | 1.03 | 4.70 |
| n=500, p=10, k=3 | Promax | 1.06 | 5.00 |
| n=1000, p=15, k=4 | No rotation | 2.57 | 7.56 |
| n=1000, p=15, k=4 | Varimax | 2.74 | 7.44 |
| n=1000, p=15, k=4 | Promax | 3.60 | 8.08 |
| n=5000, p=20, k=5 | No rotation | 7.19 | 7.78 |
| n=5000, p=20, k=5 | Varimax | 7.19 | 7.80 |
| n=5000, p=20, k=5 | Promax | 7.20 | 8.26 |

**Notes:**
- Rust is now consistently faster than R across all dataset sizes (1.1x to 18x speedup)
- Largest improvements at small/medium scale where Woodbury identity optimization is most effective
- For large datasets (n=5000), Rust is now competitive with R (~7.2 ms vs ~7.8 ms)

### Optimization Details (v2)

Key optimizations implemented:
1. **Cached correlation matrix log-determinant**: Eliminates redundant eigendecomposition per iteration
2. **Woodbury matrix identity**: For k << p, use O(k³) operations instead of O(p³) for Σ⁻¹
3. **faer-based matrix operations**: Use optimized LU decomposition for matrix inversion
4. **Parallel correlation matrix computation**: Use rayon for large datasets (n > 1000, p > 10)

## References

- Jöreskog, K. G. (1967). "Some Contributions to Maximum Likelihood Factor Analysis". *Psychometrika*, 32, 443-482.
- Jöreskog, K. G. (1969). "A General Approach to Confirmatory Maximum Likelihood Factor Analysis". *Psychometrika*, 34, 183-202.
- Kaiser, H. F. (1958). "The varimax criterion for analytic rotation in factor analysis". *Psychometrika*, 23, 187-200.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/factanal.html
