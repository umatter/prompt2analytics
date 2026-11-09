# Machine Learning Validation

This directory contains validation documentation for machine learning methods.

## Methods

Whether each method is pinned to R is in the method index of [../README.md](../README.md),
derived from `validation/methods/*.json`.

### Clustering
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| K-means | [kmeans.md](kmeans.md) | `kmeans()` | R `stats::kmeans()` |
| DBSCAN | [hclust.md](hclust.md) | `dbscan()` | R `dbscan::dbscan()` |
| Hierarchical clustering | [hclust.md](hclust.md) | `hierarchical()` | R `stats::hclust()` |
| Cut dendrogram | [cutree.md](cutree.md) | `cutree()` | R `stats::cutree()` |
| Dynamic tree cut | [dynamic_tree_cut.md](dynamic_tree_cut.md) | `run_dynamic_tree_cut()` | R `dynamicTreeCut::cutreeDynamic()` |
| HDBSCAN | [hdbscan.md](hdbscan.md) | `run_hdbscan()` | R `dbscan::hdbscan()` |
| OPTICS | [optics.md](optics.md) | `run_optics()` | R `dbscan::optics()`, `extractDBSCAN()`, `extractXi()` |
| PAM, CLARA, fuzzy c-means, FANNY | [partitioning.md](partitioning.md) | `run_kmedoids()`, `run_clara()`, `run_fuzzy_cmeans()`, `run_fanny()` | R `cluster::pam()`, `clara()`, `fanny()` |
| Affinity propagation | [affinity_propagation.md](affinity_propagation.md) | `run_affinity_propagation()` | R `apcluster::apcluster()` |
| Gaussian mixtures | [gmm.md](gmm.md) | `run_gaussian_mixture()` | R `mclust::Mclust()` |
| Mixture EMs | [mixtures.md](mixtures.md) | `run_normal_mix_em()`, `run_mvnorm_mix_em()`, `run_flexmix()` | R `mixtools::normalmixEM()`, `mvnormalmixEM()`, `flexmix::flexmix()` |
| Silhouette widths | [silhouette.md](silhouette.md) | `run_silhouette()` | R `cluster::silhouette()` |

### Dimensionality Reduction
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| PCA | [pca.md](pca.md) | `pca()` | R `stats::prcomp()` |
| Classical MDS | [cmdscale.md](cmdscale.md) | `cmdscale()` | R `stats::cmdscale()` |
| t-SNE | no validation document yet | `tsne()` | R `Rtsne` (not installed) |

### Supervised Learning
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| CART | [cart.md](cart.md) | `run_cart()` | R `rpart::rpart()` |
| Tree-ensemble impurity importance | [tree_importance.md](tree_importance.md) | `random_forest()`, `run_gbm()` | R `rpart`, `randomForest`, `gbm` |
| Random forest | [tree_importance.md](tree_importance.md) | `random_forest()` | R `randomForest::randomForest()` |
| Support vector classification | [svm.md](svm.md) | `linear_svm()`, `kernel_svm()` | R `e1071::svm()` |
| MARS | [mars.md](mars.md) | `run_mars()` | none (`earth` is not installed) |
| Projection pursuit regression | [ppr.md](ppr.md) | `run_ppr()` | R `stats::ppr()` |
| AdaBoost | [adaboost.md](adaboost.md) | `run_adaboost()` | SAMME and AdaBoost.M1 written out in R (`rpart` weak learners) |
| Model-based boosting | [mboost.md](mboost.md) | `run_mboost()` | R `mboost::glmboost()` |
| XGBoost | [xgboost.md](xgboost.md) | `run_xgboost()` | R `xgboost::xgb.train()` |
| BART | [bart.md](bart.md) | `run_bart()` | the BART model's exact posterior on unsplittable trees (BART not installed) |
| BART causal inference | [bart_causal.md](bart_causal.md) | `run_bart_causal()` | none |
| Causal forest | [causal_forest.md](causal_forest.md) | `causal_forest()` | grf (not installed) |
| C5.0 | [c50.md](c50.md) | `run_c50()` | none |
| Conditional inference trees | [ctree.md](ctree.md) | `run_ctree()` | none |
| Cubist | [cubist.md](cubist.md) | `run_cubist()` | none |

### Interpretation
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Partial dependence / ICE | [pdp.md](pdp.md) | `partial_dependence()`, `compute_ice_curves()` | R `pdp::partial()` |
| SHAP (TreeSHAP and Kernel SHAP) | [shap.md](shap.md) | `shap_tree_ensemble()`, `kernel_shap()` | exact Shapley values computed in R |

## Key Test Datasets

- **Iris**: R's `iris` (via `datasets::iris`; no committed CSV)
- **Synthetic clusters**: Known cluster assignments
- **High-dimensional data**: For dimensionality reduction

## Validation Notes

ML methods often have stochastic elements. Validation approaches:
- Use fixed random seeds where possible
- Compare clustering metrics (silhouette, ARI) rather than exact assignments
- For PCA, compare explained variance ratios
- For t-SNE, compare perplexity and basic structure

## MCP tools

The MCP tools `ml_kmeans`, `ml_dbscan`, `ml_hierarchical`, `ml_cutree`, `ml_pca`, `ml_ppr` and
`ml_mboost` are pinned to the same R cases as the core methods; each method document has an
"MCP tools" section (registry `validation/methods/mcp_ml_spatial.json`). Not pinned through the MCP:
the random-forest and SVM output columns (property tests against the core's predictions; the linear
SVM is not pinned to e1071, ML-40), SHAP (the Shapley definitions are enumerated in the core tests)
and HDBSCAN (no MCP tool).

## Running Tests

```bash
validation/run_validation.sh --rust-only --category ml
```
