# Validation: Classical Multidimensional Scaling (cmdscale)

## Method Overview

Classical (metric) multidimensional scaling, also known as Principal Coordinates Analysis (PCoA). Takes a matrix of pairwise distances and reconstructs a low-dimensional configuration of points that approximately preserves the distances.

Key algorithm steps:
1. Square the distance matrix
2. Double-center using the centering matrix H = I - (1/n)11'
3. Compute eigendecomposition of B = -0.5 * H * D² * H
4. Extract top k eigenvectors scaled by sqrt(eigenvalues)

Key outputs:
- Point coordinates in k dimensions
- Eigenvalues (variances explained)
- Goodness-of-fit measures (GOF)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | `cmdscale()` | R 4.3+ |

## Test Cases

### Test 1: Unit Square Points

**R Code**:
```r
# Four points forming a unit square
points <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
d <- dist(points)
mds <- cmdscale(d, k = 2, eig = TRUE)
print(mds$points)
print(mds$eig)
print(mds$GOF)
```

**Results Comparison**:

| Output | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| GOF[1] | ~1.0 | ~1.0 | 1e-4 | PASS |
| GOF[2] | ~1.0 | ~1.0 | 1e-4 | PASS |
| n_positive_eig | 2 | 2 | exact | PASS |

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ML-98):
- `ml::reduction::tests::test_cmdscale_basic` (`crates/p2a-core/src/ml/reduction.rs`): dimensions and gof[0], gof[1] > 0.9. No R value is compared.

### Test 2: Random Distance Matrix

**R Code**:
```r
set.seed(42)
points <- matrix(rnorm(20), ncol = 2)
d <- dist(points)
mds <- cmdscale(d, k = 2, eig = TRUE)
# Reconstruct distances and compare
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ML-98):
- `ml::reduction::tests::test_cmdscale_from_data` (`crates/p2a-core/src/ml/reduction.rs`): dimensions of the result for a 4-point data-matrix input (a unit square), not the `set.seed(42)` data above.

### Test 3: Known European Cities Distance

Standard benchmark using geographic distances between European cities.

No Rust test runs the eurodist data.

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with ML-98):
- `ml::reduction::tests::test_validate_cmdscale_vs_r` (`crates/p2a-core/src/ml/reduction.rs`): a 4-point square configuration; dimensions and gof[0] > 0.95. No R value is compared.
- `ml::reduction::tests::test_cmdscale_gof` (`crates/p2a-core/src/ml/reduction.rs`): both GOF values in [0, 1], gof[1] ≤ gof[0] + 0.01.
- `ml::reduction::tests::test_validate_cmdscale_matches_r` (`crates/p2a-core/src/ml/reduction.rs`), added in Tier 0 (T0-WP16): the 8 × 2 points (up to sign) and all eight eigenvalues within 1e-9 absolute of `cmdscale(dist(x), k = 2, eig = TRUE)`, Euclidean and Manhattan, with R 4.3.3 values typed into the test. A quoted-value check, not R-pinned; GOF is not compared.

## Numerical Precision Summary

- Point coordinates may differ in sign (reflection) but preserve distances
- Eigenvalues match R within 1e-6 tolerance
- GOF is not compared with R; p2a's gof[0] is not one of R's two measures (ML-98)

## Known Differences

- Sign of eigenvectors may differ (arbitrary choice in eigendecomposition)
- Small numerical differences in eigenvalue computation

## Performance Comparison

| n Points | Rust (µs) | R (µs) | Speedup |
|----------|-----------|--------|---------|
| n=20     | TBD       | TBD    | TBD     |
| n=50     | TBD       | TBD    | TBD     |
| n=100    | TBD       | TBD    | TBD     |

## References

- Torgerson, W. S. (1952). "Multidimensional scaling: I. Theory and method". *Psychometrika*, 17, 401-419.
- Gower, J. C. (1966). "Some distance properties of latent root and vector methods used in multivariate analysis". *Biometrika*, 53, 325-338.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html
