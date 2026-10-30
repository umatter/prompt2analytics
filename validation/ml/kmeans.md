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

## Missing and non-finite values

- **Core.** `kmeans` refuses NaN and ±Inf before it initialises and names the first bad
  cell in row-major order: "k-means: x must contain only finite values (row 4, column 0 is
  NaN); remove or impute missing values first". It used to report "K-means failed to
  converge" (every inertia was NaN). R stops as well: `kmeans(m, 2)` on a matrix with an
  `NA` gives "NA/NaN/Inf in foreign function call (arg 1)" (R 4.3.3).
- **MCP and CLI.** `ml_kmeans` and `p2a ml kmeans` read their columns through the shared
  extractor with the Fail policy: a missing value (null or NaN) is an error naming the
  column ("Column 'x1_na' contains 1 missing value(s) (null or NaN) ..."; the MCP text adds
  that `munge_drop_na` or `munge_fill_na` can remove or impute the rows), a text column is
  refused by name, and integer and boolean columns are numeric.
- **Tests.** `test_na_kmeans_refuses_non_finite_input` (`crates/p2a-core/src/ml/clustering.rs`),
  `test_na_mcp_ml_matrix_tools_name_the_column` and
  `test_mcp_ml_matrix_tools_int64_equals_float64` (`crates/p2a-mcp/src/tools/handlers/ml.rs`),
  `test_cli_ml_integer_columns_accepted` (`crates/p2a-cli/tests/tier2/wp17_ml.rs`). They
  compare p2a with p2a and pin no R number.

## Running the Tests

```bash
cargo test -p p2a-core -- kmeans
```

## References

- Lloyd, S.P. (1982). "Least Squares Quantization in PCM". *IEEE Transactions on Information Theory*, 28(2), 129-137.
