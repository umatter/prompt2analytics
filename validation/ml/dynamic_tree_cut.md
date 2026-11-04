# Validation: dynamic tree cut (dynamicTreeCut::cutreeDynamic)

## Method

`p2a_core::ml::advanced_clustering_mod::dynamic_tree_cut` ports dynamicTreeCut 1.63.1 (pure R), default
paths:

- **tree** (`cutreeDynamicTree`): a static cut at `cutHeight` (default 99% of the largest height), then each
  branch is split by the shape of its leaf-order height profile (`.processIndividualCluster`), once
  (`deep_split = 0`, R's `deepSplit = FALSE`) or until no branch splits further (`deep_split >= 1`, TRUE).
  Modules smaller than `min_cluster_size` stay unassigned (label 0). Needs non-decreasing heights, as R's
  `cutree(h =)`.
- **hybrid** (`cutreeHybrid`): branches are built bottom-up below `cutHeight` and kept apart when they pass
  the core-scatter, gap and size criteria (`deepSplit` 0-4 selects `maxCoreScatter` 0.64-0.95 and
  `minGap = (1 - maxCoreScatter) * 3/4`); then the PAM stage assigns small clusters and unassigned objects on
  a composite branch to the nearest cluster of that branch by mean dissimilarity, when it is below the cluster
  diameter or `maxPamDist`.

Fixed at dynamicTreeCut's defaults: `pamRespectsDendro = TRUE`, `useMedoids = FALSE`,
`respectSmallClusters = TRUE`, no external split functions. As `cutreeDynamic`: the default method is hybrid,
`minClusterSize` 20, `deepSplit` 1 for hybrid and FALSE for tree, `maxPamDist` = `cutHeight` as given (else the
computed cut height). Hybrid without dissimilarities runs the tree method and warns ("method 'hybrid' requires a
dissimilarity matrix (distM); defaulting to method 'tree'"), as R does. `min_cluster_size` must be at least 2
(dynamicTreeCut fails on one-object clusters); n < 2 is an error; n = 2 gives R's all-unassigned result (0 0).

`cutree_dynamic` reads the dissimilarities through `DynamicCutDistances` (a square matrix or the condensed
triangle, no copy); `run_dynamic_tree_cut` builds the tree with `fastcluster` and passes the condensed
Euclidean dissimilarities.

Labels: 0 unassigned, clusters 1.. by decreasing size (ties by the earlier cluster), as R.

## The two documented differences

- **No module.** When the tree method leaves no module of `min_cluster_size` objects, p2a returns all zeros;
  R's `.assignModuleNumber` returns NA labels.
- **Nearest colour 0.** In the PAM stage the candidate clusters of a composite branch include the unassigned
  group (colour 0) when one of its basic branches is not a cluster. When that group is the nearest, p2a leaves
  the objects unassigned; R stops with "argument is of length zero".

Means and column sums are taken in f64 (R accumulates them in long double).

## R code

`validation/r/cases/ml_hierarchical.R`, case `dynamic_tree_cut` (dynamicTreeCut 1.63.1, R 4.3.3), on
`hclust(dist(X), "average")` of `validation/datasets/generated/hclust_60.csv` with `minClusterSize = 5` unless
stated.

## Values

| Key | Call | Clusters | Unassigned | Sizes |
|---|---|---|---|---|
| `tree_deep_false` | tree, deepSplit FALSE | 4 | 4 | 20, 16, 14, 6 |
| `tree_deep_true` | tree, deepSplit TRUE | 7 | 4 | 16, 10, 9, 6, 5, 5, 5 |
| `tree_deep_true_cut15` | tree, TRUE, cutHeight 1.5 | 3 | 35 | 14, 6, 5 |
| `hybrid_ds0` .. `hybrid_ds4` | hybrid, deepSplit 0-4 | 3, 4, 5, 5, 6 | 0 | ds2 20, 17, 13, 5, 5 |
| `hybrid_ds2_nopam` | hybrid, 2, pamStage FALSE | 5 | 1 | 20, 16, 13, 5, 5 |
| `hybrid_ds2_cut15` | hybrid, 2, cutHeight 1.5 | 4 | 35 | 9, 6, 5, 5 |
| `hybrid_ds2_maxpam0` | hybrid, 2, maxPamDist 0 | 5 | 1 | 20, 16, 13, 5, 5 |
| `hybrid_defaults` | hybrid, every default (minClusterSize 20) | 2 | 0 | 40, 20 |
| `ward_hybrid_ds1` | `hclust(dist(X), "ward.D2")`, hybrid, 1 | 4 | 0 | 22, 20, 12, 6 |

Hand cases (R): six points in two groups, ward.D2, tree, deepSplit TRUE, minClusterSize 2: `2 2 2 1 1 1`; six
points on a line, ward.D2, hybrid, minClusterSize 2: deepSplit 0 and 4 both `1 1 1 2 2 2`.

## Tests

| Test (`crates/p2a-core/src/ml/advanced_clustering_mod/dynamic_tree_cut.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_cutree_dynamic_tree` | `hclust_average_height`, `tree_*` | heights `Tol::coef(60)`; labels exact |
| `test_validate_cutree_dynamic_hybrid` | `hybrid_*`, `ward_hybrid_ds1` (square and condensed dissimilarities agree) | exact |
| `test_validate_dynamic_tree_cut_options_change_the_result` | every exposed option changes the labels, as in R | property |
| `test_validate_dynamic_tree_cut_defaults_and_fallback` | defaults, the warned fallback to tree, the option checks | property |
| `test_dynamic_tree_cut_n_below_2_is_error`, `test_dynamic_tree_cut_n2_unassigned_as_r`, `test_dynamic_tree_cut_basic`, `test_dynamic_tree_cut_deep_split` | hand cases above | exact |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::advanced_clustering_mod::dynamic_tree_cut
```

## References

- Langfelder, P., Zhang, B., and Horvath, S. (2008). "Defining clusters from a hierarchical cluster tree: the
  Dynamic Tree Cut package for R." *Bioinformatics*, 24(5), 719-720.
- dynamicTreeCut 1.63.1, `cutreeDynamic`, `cutreeDynamicTree`, `cutreeHybrid`.
