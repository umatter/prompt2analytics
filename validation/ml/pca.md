# Validation: Principal Component Analysis (PCA)

## Method Overview

PCA finds orthogonal directions (principal components) that maximize variance in the data.

**Computation**:
1. Center data: X_centered = X - mean(X)
2. Compute covariance matrix: C = X'X / (n-1)
3. Eigendecomposition: C = VΛV'
4. PCs are columns of V, sorted by eigenvalues

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `prcomp()` | 4.3.x |
| scikit-learn | Python | `PCA()` | 1.3-x |

## Test Cases

### Test 1: Simple 2D Data

**R Code**:
```r
set.seed(42)
n <- 100

# Correlated 2D data
x1 <- rnorm(n)
x2 <- 0.8 * x1 + rnorm(n, 0, 0.5)
data <- cbind(x1, x2)

pca_result <- prcomp(data, center = TRUE, scale = FALSE)

# Loadings
print(pca_result$rotation)

# Variance explained
summary(pca_result)
```

**Results Comparison**:

| Statistic | R prcomp | p2a Rust | Tolerance |
|-----------|----------|----------|-----------|
| PC1 loading | varies | varies | 0.01 |
| % Variance PC1 | ~85% | ~85% | 2% |

---

### Test 2: Iris Dataset

**R Code**:
```r
data(iris)
pca_result <- prcomp(iris[, 1:4], center = TRUE, scale = TRUE)
summary(pca_result)

# First 2 PCs explain ~95% of variance
```

## Sign Ambiguity

Note: PCA loadings are unique up to sign. Compare absolute values or check correlation.

## Running the Tests

```bash
cargo test -p p2a-core -- pca
```

## References

- Jolliffe, I.T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
