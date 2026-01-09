# Machine Learning Validation

This directory contains validation documentation for machine learning methods.

## Methods

### Clustering
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| K-means | [kmeans.md](kmeans.md) | `kmeans()` | R `stats::kmeans()`, sklearn |
| DBSCAN | [dbscan.md](dbscan.md) | `dbscan()` | sklearn, R `dbscan` |
| Hierarchical | [hierarchical.md](hierarchical.md) | `hierarchical()` | R `hclust()`, scipy |

### Dimensionality Reduction
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| PCA | [pca.md](pca.md) | `pca()` | R `prcomp()`, sklearn |
| t-SNE | [tsne.md](tsne.md) | `tsne()` | sklearn, R `Rtsne` |

### Supervised Learning
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Random Forest | [random_forest.md](random_forest.md) | `random_forest()` | sklearn, R `randomForest` |
| SVM | [svm.md](svm.md) | `linear_svm()` | sklearn, R `e1071` |

## Key Test Datasets

- **Iris**: Classic classification dataset (n=150, k=4)
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
cargo test -p p2a-core -- ml::tests::test_validate
```
