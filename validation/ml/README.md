# Machine Learning Validation

This directory contains validation documentation for machine learning methods.

## Methods

### Clustering
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| K-means | [kmeans.md](kmeans.md) | `kmeans()` | R `stats::kmeans()`, sklearn |
| DBSCAN | no validation document yet | `dbscan()` | sklearn, R `dbscan` |
| Hierarchical | no validation document yet | `hierarchical()` | R `hclust()`, scipy |

### Dimensionality Reduction
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| PCA | [pca.md](pca.md) | `pca()` | R `prcomp()`, sklearn |
| t-SNE | no validation document yet | `tsne()` | sklearn, R `Rtsne` |

### Supervised Learning
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Random Forest | no validation document yet | `random_forest()` | sklearn, R `randomForest` |
| SVM | no validation document yet | `linear_svm()` | sklearn, R `e1071` |

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

## Running Tests

```bash
validation/run_validation.sh --rust-only --category ml
```
