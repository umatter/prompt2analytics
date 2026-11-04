# Validation: hierarchical clustering core (stats::hclust, stats::cutree)

## Method Overview

`p2a_core::ml::hclust` (`crates/p2a-core/src/ml/hclust.rs`) is the one agglomerative core of p2a. It
implements the nearest-neighbour-list algorithm of R's `hclust.f` (`HCLUST`, Murtagh 1983/1985, with J. B.'s
fix for non-monotone methods) and the merge-matrix and order conventions of `HCASS2`:

- ascending scans with strict `<`, so on exactly tied dissimilarities the first minimal pair is merged, as in R;
- the Lance-Williams update of every active cluster's dissimilarity to the merged cluster, in `hclust.f`'s
  operand order;
- the eight methods of `hclust`: `single`, `complete`, `average` (UPGMA), `mcquitty` (WPGMA, "weighted"),
  `ward.D`, `ward.D2`, `centroid` (UPGMC), `median` (WPGMC, "gower");
- centroid and median on squared Euclidean distances, as `?hclust` documents (`hclust_euclidean` squares
  `dist(x)` for them);
- **"ward" means ward.D2** in all of p2a (`Linkage::Ward`; `cluster::agnes(method = "ward")`,
  `fastcluster::hclust.vector(method = "ward")`). R's own deprecated `hclust(method = "ward")` is ward.D
  (`Linkage::WardD`, spellings `ward.D`, `ward1`).

Memory is one condensed triangle of n(n-1)/2 doubles; time is typically O(n^2). `cutree` numbers groups by first
appearance in observation order, as R; a cut by height is refused for a tree whose heights decrease (centroid,
median), as R's `cutree(h =)` refuses it. Non-finite data and distances that overflow f64 (also after squaring,
for ward.D2, centroid and median) are refused.

## Reference

`stats::hclust` and `stats::cutree` (R 4.3.3).

## Where

R code `validation/r/cases/ml_hierarchical.R`, case `hclust_methods`, datasets
`validation/datasets/generated/hclust_60.csv` (60 x 3, three Gaussian groups of 20, seed 20261110) and
`validation/datasets/generated/hclust_ties.csv` (40 points on a 0.5 grid, seed 7, with exactly tied distances).

## Tests and tolerances

| Test | Keys | Tolerance |
|---|---|---|
| `test_validate_hclust_single`, `_complete`, `_average`, `_mcquitty`, `_wardD`, `_wardD2`, `_centroid`, `_median` | `<method>_merge`, `_height`, `_order`, `_cutree3`; `ties_<method>_merge`, `_height`, `_order` | merge, order, labels exact; heights `Tol::coef(n)` |
| `test_validate_hclust_cuts` | `average_cutree_k246`, `average_cutree_h15`, `average_k_h15` | exact |
| `test_validate_hclust_hand_cases` | `ml13_*`, `ml14_*`, `ml25_wardD2_height` | heights `Tol::coef(n)`; labels exact |
| `test_validate_hclust_tree_conversions_round_trip` | (properties on `hclust_methods` data: merge-matrix, SciPy-linkage and order round trips) | exact |
