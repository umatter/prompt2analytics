# Validation: Cut Dendrogram (cutree)

## Method

`cutree`, `cutree_multiple_k` and `run_cutree` (`crates/p2a-core/src/ml/clustering.rs`) cut the tree of
`hierarchical` with the hclust core's `cut_k` / `cut_height` (`crates/p2a-core/src/ml/hclust.rs`), R's
`cutree(tree, k =)` and `cutree(tree, h =)`:

- groups are numbered by first appearance in observation order, as R (labels are R's minus 1);
- `cutree(h =)` takes R's rule for the number of groups and is refused for a tree whose heights decrease
  (centroid, median), as R refuses it;
- the linkage matrix must hold the complete tree (n - 1 merges): a truncated one is refused ("cutree needs the
  complete tree: ...").

## R code

`validation/r/cases/ml_hierarchical.R`, case `hclust_methods` (see `validation/ml/hclust.md`):
`cutree(hclust(dist(X), m), 3)` for the eight methods, `cutree(, k = c(2, 4, 6))` and `cutree(, h = 1.5)` of
the average tree.

## Values

`cutree(, 3)` group sizes on hclust_60: single 39, 1, 20; complete 27, 13, 20; average, ward.D, ward.D2,
centroid 22, 18, 20; mcquitty, median 21, 19, 20. The review's four hand cases (ML-14) are in
`hclust.md`.

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/clustering.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_cutree3` | `<method>_cutree3`, `average_cutree_k246`, `average_cutree_h15`, `average_k_h15` | exact |
| `test_validate_cutree_labels_deterministic_first_appearance` | (ward.D2 tree of hclust_60 cut repeatedly: identical labels, the first observation in group 1) | property (data from case `hclust_methods`), not an R comparison; exact |
| `test_cutree_refuses_a_truncated_tree`, `test_hierarchical_height_cut_refuses_inversions` | error texts | |
| `test_cutree_basic`, `test_cutree_by_height`, `test_cutree_multiple_k`, `test_cutree_validation` | p2a properties | |

## Missing and non-finite values

- **Core.** `hierarchical` (which `cutree` cuts) refuses NaN and ±Inf and names the first
  bad cell in row-major order: "hierarchical clustering: data must contain only finite values
  (row 4, column 0 is NaN); remove or impute missing values first". It used to return a
  corrupt tree: a NaN row left every nearest-neighbour distance at infinity and the merge
  loop merged anyway (self-merges, more clusters than requested). The merge loop also
  returns an error instead of merging when no finite distance is left, which finite data
  reach when their distances overflow f64 (values around 1e200): "no finite distance
  remains between the clusters still to be merged ...". Distances that overflow f64 (also
  after squaring, for Ward, centroid and median) are refused in both modes, with or without
  `distance_threshold`; rescale the columns.
- **R.** `dist()` uses the coordinates both rows have (scaled up), so
  `hclust(dist(m), "ward.D2")` clusters a row with one `NA` (heights 1, 1, 1.5275252316519468,
  10.220893633467998 on the review's rows (0,0), (1,0), (5,5), (6,5), (NA,1); R 4.3.3).
  p2a does not implement these pairwise-complete distances and refuses the row instead (its
  missing-value rule for ML methods is Fail): a documented deviation. With infinite distances
  (`dist()` of (0,0), (1e200,0), (2e200,0) is Inf Inf Inf), `hclust` stops with "NA/NaN/Inf in
  foreign function call (arg 10)", as p2a does.
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
