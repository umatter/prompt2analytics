# Validation: Cut Dendrogram (cutree)

## Method Overview

Cuts a hierarchical clustering dendrogram into groups. Given the output of `hclust()`, this function assigns each observation to a cluster based on either:
- A specified number of clusters (k)
- A specified height cutoff (h)

Key outputs:
- Cluster labels (1 to k) for each observation
- Optional: labels for multiple k values

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats   | R        | `cutree()` | R 4.3+ |

## Test Cases

### Test 1: Simple 6-Point Clustering

**R Code**:
```r
# Six points in two distinct clusters
points <- matrix(c(1, 1.1, 1.2, 5, 5.1, 5.2,
                   1, 1.1, 1.2, 5, 5.1, 5.2), ncol = 2)
hc <- hclust(dist(points), method = "complete")
ct <- cutree(hc, k = 2)
print(ct)
```

**Results Comparison**:

| Output | R Value | Rust Value | Tolerance | Status |
|--------|---------|------------|-----------|--------|
| labels[0:2] | 1, 1, 1 | 1, 1, 1 | exact | PASS |
| labels[3:5] | 2, 2, 2 | 2, 2, 2 | exact | PASS |

**Rust Test**: `crates/p2a-core/src/ml/clustering.rs::tests::test_cutree_basic`

### Test 2: Cut by Height

**R Code**:
```r
# Cut at a specific height
hc <- hclust(dist(matrix(rnorm(20), ncol=2)), method = "complete")
ct <- cutree(hc, h = 1.5)
print(ct)
```

**Rust Test**: `crates/p2a-core/src/ml/clustering.rs::tests::test_cutree_by_height`

### Test 3: Multiple k Values

**R Code**:
```r
# Get cluster assignments for k=2, 3, 4
hc <- hclust(dist(matrix(rnorm(20), ncol=2)), method = "complete")
ct <- cutree(hc, k = 2:4)
print(ct)  # Matrix with columns for each k
```

**Rust Test**: `crates/p2a-core/src/ml/clustering.rs::tests::test_cutree_multiple_k`

## Numerical Precision Summary

- Cluster labels match R exactly
- The algorithm uses union-find with path compression for efficiency

## Known Differences

- None identified for same input data

## Performance Comparison

| n Points | Rust (µs) | R (µs) | Speedup |
|----------|-----------|--------|---------|
| n=50     | TBD       | TBD    | TBD     |
| n=100    | TBD       | TBD    | TBD     |
| n=200    | TBD       | TBD    | TBD     |

## References

- Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cutree.html
