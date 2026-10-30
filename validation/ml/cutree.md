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

## Missing and non-finite values

- **Core.** `hierarchical` (which `cutree` cuts) refuses NaN and ±Inf and names the first
  bad cell in row-major order: "hierarchical clustering: x must contain only finite values
  (row 4, column 0 is NaN); remove or impute missing values first". It used to return a
  corrupt tree: a NaN row left every nearest-neighbour distance at infinity and the merge
  loop merged anyway (self-merges, more clusters than requested). The merge loop also
  returns an error instead of merging when no finite distance is left, which finite data
  reach when their distances overflow f64 (values around 1e200): "no finite distance
  remains between the clusters still to be merged ...".
- **R.** `dist()` uses the coordinates both rows have (scaled up), so
  `hclust(dist(m), "ward.D2")` clusters a row with one `NA` (heights 1, 1, 1.5275252316519468,
  10.220893633467998 on the review's rows (0,0), (1,0), (5,5), (6,5), (NA,1); R 4.3.3).
  p2a does not implement these pairwise-complete distances and refuses the row instead (T2
  design 1.6, Fail): a documented deviation. With infinite distances (`dist()` of
  (0,0), (1e200,0), (2e200,0) is Inf Inf Inf), `hclust` stops with "NA/NaN/Inf in foreign
  function call (arg 10)", as p2a does.
- **MCP and CLI.** `ml_hierarchical`, `ml_cutree` and `p2a ml hierarchical` read their
  columns with the Fail policy: a missing value is an error naming the column, and integer
  columns are numeric.
- **Tests.** `test_na_hierarchical_refuses_non_finite_input` and
  `test_hierarchical_errors_when_no_finite_distance_remains`
  (`crates/p2a-core/src/ml/clustering.rs`), `test_na_mcp_ml_matrix_tools_name_the_column`
  and `test_mcp_ml_matrix_tools_int64_equals_float64`
  (`crates/p2a-mcp/src/tools/handlers/ml.rs`), `test_na_cli_ml_missing_values_refused`
  (`crates/p2a-cli/tests/tier2/wp17_ml.rs`). They compare p2a with p2a and pin no R number.

## References

- Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- R stats package documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cutree.html
