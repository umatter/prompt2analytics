# Validation: hierarchical clustering core (stats::hclust, stats::cutree)

## Method

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

Height scales follow R per entry point: `hierarchical`, AGNES and pvclust report the heights of
`hclust(dist(x)^2, m)` for centroid and median; `fastcluster` reports their square roots, as
`fastcluster::hclust.vector` does.

## R code

`validation/r/cases/ml_hierarchical.R`, case `hclust_methods` (R 4.3.3, cluster 2.1.6), datasets
`validation/datasets/generated/hclust_60.csv` (60 x 3, three Gaussian groups of 20, seed 20261110) and
`validation/datasets/generated/hclust_ties.csv` (40 points on a 0.5 grid, seed 7, with exactly tied distances).

## Values

`hclust(dist(X), m)` on hclust_60 (centroid and median on `dist(X)^2`):

| Method | First two heights | Last two heights | `cutree(, 3)` sizes |
|---|---|---|---|
| single | 0.23993038929926197, 0.28126955844928964 | 2.0240647724646115, 2.701454020831262 | 39, 1, 20 |
| complete | same | 8.144349741528764, 10.304529754773773 | 27, 13, 20 |
| average | same | 4.756374839641573, 5.7323973009658165 | 22, 18, 20 |
| mcquitty | same | 6.121793095393411, 6.655335493459106 | 21, 19, 20 |
| ward.D | same | 48.390725614722854, 87.1861152476682 | 22, 18, 20 |
| ward.D2 | same | 18.923700218976915, 25.991679261603267 | 22, 18, 20 |
| centroid | 0.0575665917092954, 0.07911256451025836 | 18.086183332207426, 25.333777156427146 | 22, 18, 20 |
| median | same as centroid | 31.83876739289265, 35.6067575140773 | 21, 19, 20 |

Ties: on hclust_ties every method's merge, order and heights equal R's exactly (`ties_<method>_*`); the
tie order falls out of the ascending strict-`<` scans, so no property-test fallback was needed.

Hand cases (review findings): `{0, 1, 3, 7, 15}` average heights 1, 2.5, 5.666666666666667, 12.25 and mcquitty
1, 2.5, 5.25, 10.625 (ML-13); the four `cutree` cases of ML-14 (`ml14_*`); ward.D2 heights (ML-25).

## Wrappers

- **`hierarchical`** (`ml::clustering`): the core on `dist(x)`, cut by `n_clusters` or `distance_threshold`
  (`cutree(h =)`); labels are `cutree - 1`; `linkage_matrix` is the SciPy form of the same tree.
- **`cutree`, `cutree_multiple_k`**: R's `cutree(k =)` on a complete tree; a truncated linkage matrix is refused.
- **AGNES** (`advanced_clustering_mod::hierarchical::agnes`): the full tree from the core, cut for the labels.
  `ac` is cluster's `bncoef` from first-merge heights and does not depend on `n_clusters`: average
  0.8259032898863274, single 0.6885297065427618, complete 0.8984982990829474, ward 0.9590527579487206, weighted
  0.844242183861284. `order` is agnes' banner order. Tie note: on exactly tied dissimilarities the core merges
  the first tied pair (as `stats::hclust`), `cluster::agnes` the last, so tied merges can come in another order
  than R's `agnes` (registry deviation ML-55).
- **DIANA** (`diana`): cluster's `splyt` with running sums; the full divisive tree is always built, so `dc`
  (0.8885450615416582), the banner, `order` and `merge` equal `cluster::diana`; labels are
  `cutree(as.hclust(diana(x)), k) - 1`.
- **pvclust** (`pvclust`): the original tree is `hclust(dist(x), "average")` (merge and heights pinned); the
  bootstrap p-values are not R-pinned (ML-10, Tier 4). An unknown method name is an error.
- **fastcluster** (`fastcluster`): the core with `fastcluster::hclust` / `hclust.vector` conventions (`Weighted`
  is mcquitty, `Ward` is ward.D2, centroid and median heights square-rooted).

## DBSCAN

There is no `validation/ml/dbscan.md`; the DBSCAN pin uses the same data. `dbscan::dbscan(X, eps = 1, minPts =
5)` (dbscan 1.2.6) on hclust_60, case `kmeans_lloyd_pca_mds`: 3 clusters of 9, 5 and 16 points and 30 noise
points; p2a's label + 1 equals R's cluster (noise -1 -> 0). Test `test_validate_dbscan_hclust60`
(`crates/p2a-core/src/ml/clustering.rs`), exact.

## Tests and tolerances

| Test | Keys | Tolerance |
|---|---|---|
| `test_validate_hclust_single`, `_complete`, `_average`, `_mcquitty`, `_wardD`, `_wardD2`, `_centroid`, `_median` | `<method>_merge`, `_height`, `_order`, `_cutree3`; `ties_<method>_merge`, `_height`, `_order` | merge, order, labels exact; heights `Tol::coef(n)` |
| `test_validate_hclust_cuts` | `average_cutree_k246`, `average_cutree_h15`, `average_k_h15` | exact |
| `test_validate_hclust_hand_cases` | `ml13_*`, `ml14_*`, `ml25_wardD2_height` | heights `Tol::coef(n)`; labels exact |
| `test_validate_hclust_tree_conversions_round_trip` | (properties on `hclust_methods` data: merge-matrix, SciPy-linkage and order round trips) | exact |
| `test_validate_hierarchical_linkages` (`ml/clustering.rs`) | `<method>_merge`, `_height`; `ties_<method>_merge`, `_height` | merge exact; heights `Tol::coef(n)` |
| `test_validate_cutree3`, `test_validate_cutree_labels_deterministic_first_appearance` (`ml/clustering.rs`) | `<method>_cutree3`, `average_cutree_k246`, `average_cutree_h15`, `average_k_h15` | exact |
| `test_validate_fastcluster_linkages`, `test_validate_fastcluster_finding_cases` (`advanced_clustering_mod/hierarchical.rs`) | `<method>_*`, `ties_<method>_*`, `ml13_*`, `ml14_*` | merge, order, labels exact; heights `Tol::coef(n)` |
| `test_validate_agnes_diana_coefficients`, `test_validate_agnes_tree` | `agnes_<method>_ac`, `_height`, `_order`, `_cutree3`; `diana_dc` | ac, dc `Tol::stat(60)`; heights `Tol::coef(60)`; order, labels exact |
| `test_validate_diana_banner` | `diana_banner`, `_order`, `_merge`, `_cutree3` | banner `Tol::coef(60)`; the rest exact |
| `test_validate_pvclust_tree` | `average_merge`, `average_height` | merge exact; heights `Tol::coef(60)` |
| `test_validate_dbscan_hclust60` (`ml/clustering.rs`) | `dbscan_eps1_minpts5_cluster` | exact |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::hclust ml::clustering \
    ml::advanced_clustering_mod::hierarchical
```

## References

- Murtagh, F. (1985). *Multidimensional Clustering Algorithms*. COMPSTAT Lectures 4. Physica-Verlag.
- Kaufman, L. and Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
- R `stats::hclust`, `stats::cutree`; `cluster::agnes`, `cluster::diana`; `fastcluster::hclust`.
