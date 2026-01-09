# Validation: K-Means Clustering

## Method Overview

K-means partitions data into K clusters by minimizing within-cluster variance.

**Algorithm**:
1. Initialize K centroids
2. Assign points to nearest centroid
3. Update centroids as cluster means
4. Repeat until convergence

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `kmeans()` | 4.3.x |
| scikit-learn | Python | `KMeans()` | 1.3-x |

## Test Cases

### Test 1: Well-Separated Clusters

**R Code**:
```r
set.seed(42)

# Three well-separated clusters
n <- 150
cluster1 <- matrix(rnorm(n/3 * 2, mean = 0), ncol = 2)
cluster2 <- matrix(rnorm(n/3 * 2, mean = 5), ncol = 2)
cluster3 <- matrix(rnorm(n/3 * 2, mean = c(0, 5)), ncol = 2, byrow = TRUE)

data <- rbind(cluster1, cluster2, cluster3)

# K-means with K=3
result <- kmeans(data, centers = 3, nstart = 10)
print(result$centers)
print(result$withinss)
```

**Validation Criteria**:
- All points correctly assigned
- Centroids close to true means
- Within-cluster SS minimized

---

### Test 2: Iris Dataset

**R Code**:
```r
data(iris)
result <- kmeans(iris[, 1:4], centers = 3, nstart = 25)

# Compare to true species
table(result$cluster, iris$Species)
```

## Numerical Precision

Centroids should match within 1e-4 (given same initialization).

## Running the Tests

```bash
cargo test -p p2a-core -- kmeans
```

## References

- Lloyd, S.P. (1982). "Least Squares Quantization in PCM". *IEEE Transactions on Information Theory*, 28(2), 129-137.
