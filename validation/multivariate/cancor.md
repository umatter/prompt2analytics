# Validation: Canonical Correlation Analysis (CCA)

## Method Overview

Canonical Correlation Analysis finds linear combinations of two sets of variables that have maximum correlation with each other. Given X (n × p) and Y (n × q), it computes:
- Canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρᵣ where r = min(p, q)
- X coefficients (xcoef): p × r matrix for linear combinations of X
- Y coefficients (ycoef): q × r matrix for linear combinations of Y

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | cancor() | R 4.3+ |

## Algorithm

The implementation uses a numerically stable approach:

1. Center X and Y by subtracting column means
2. Compute covariance matrices: Σxx, Σyy, Σxy
3. Cholesky decompose: Lx = chol(Σxx), Ly = chol(Σyy)
4. Form: M = Lx⁻ᵀ Σxy Ly⁻¹
5. SVD: M = U Σ Vᵀ
6. Canonical correlations = singular values
7. Coefficients: xcoef = Lx⁻¹ U, ycoef = Ly⁻¹ V

## Test Cases

### Case `stat_cancor_t3` (R-pinned in part)

Case in `validation/r/cases/stats_misc_t3.R`, dataset `validation/datasets/generated/cancor_t3.csv`
(n = 100).

| Quantity | R | Compared |
|---|---|---|
| `xcenter` | -0.011227038200000006, 0.020827766600000002, 0.0393129329 | `Tol::coef(100)` |
| `ycenter` | -0.017868867900000005, 0.024178476800000005 | `Tol::coef(100)` |
| number of canonical pairs | 2 | exact |
| `cor` | 0.8793290724715916, 0.13269379517114782 | no (STAT-16) |
| `xcoef`, `ycoef` | in the case, sign-normalised | no (STAT-16) |

**Rust tests:** `stats::cancor::tests::test_validate_cancor_t3_centres` (registry
`validation/methods/stats_misc_t3.json`, id `cancor_t3`);
`stats::cancor::tests::test_cancor_dimensions_and_order` (dimensions and order, not R-pinned).

## Known Differences

0. **STAT-16:** the canonical correlations and coefficients are not R's: p2a uses Sigma^-1 where R uses the Cholesky factor. They are registry deviations, not compared.

1. **Coefficient signs**: The sign of canonical coefficients may be flipped compared to R. This is mathematically equivalent since both (a, b) and (-a, -b) produce the same correlation.

2. **Near-singular matrices**: Our implementation uses Cholesky decomposition which may fail for singular covariance matrices. R's implementation uses QR decomposition which can handle some rank-deficient cases.

3. **Centering**: Our implementation centers data by default (xcenter=true, ycenter=true), matching R's default behavior.

4. **Missing values**: `run_cancor` refuses a null or NaN cell with `NullValues` naming the column, as R's `cancor` refuses NA; integer columns are accepted (T2-WP06; `stats::cancor::tests::test_na_cancor_columns`). Not R-pinned (STAT-16).

## Performance Comparison

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 25        | 196    | **7.9x** ✅ |
| n=1,000      | 20,135    | 595    | 0.03x ⚠️ |
| n=10,000     | 126,680   | 2,312  | 0.02x ⚠️ |
| n=100,000    | 580,740   | 29,986 | 0.05x ⚠️ |

*Benchmarked 2026-01-20. Rust performs well for small samples but R's BLAS/LAPACK backend dominates for large n.*

### Performance Notes

For **small samples (n ≤ 100)**, the Rust implementation is significantly faster than R due to reduced overhead. This covers many practical use cases in canonical correlation analysis where sample sizes are typically small to moderate.

For **large samples (n > 1000)**, R's implementation benefits from highly optimized BLAS/LAPACK routines for the X'X matrix multiplication which dominates the computation time. The Rust implementation uses faer's matrix operations which, while correct, do not achieve the same level of optimization for this particular operation pattern.

**Recommendations:**
- For typical CCA use cases with n < 500, use the Rust implementation
- For large-scale analyses (n > 10,000), consider the performance tradeoff

## References

- Hotelling, H. (1936). "Relations Between Two Sets of Variates". *Biometrika*, 28(3/4), 321-377.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cancor.html
- Gundersen, G. (2018). "Canonical Correlation Analysis". https://gregorygundersen.com/blog/2018/07/17/cca/
