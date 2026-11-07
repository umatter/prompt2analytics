# Impurity importance of the tree ensembles

## Definition

Each accepted split adds its SSE decrease, on the node's rows with their bootstrap multiplicity, to the feature it
splits on. Case file: `validation/r/cases/ml_tree_importance.R`. Registry: `validation/methods/ml_cart_cp.json`.

- **Random forest** (`RandomForestResult::inc_node_purity`): the mean over trees of the in-bag SSE decrease, which is
  randomForest's `IncNodePurity`. `feature_importances` is each feature's share.
- **Quantile forest** (`QuantileRfResult::inc_node_purity`): the same quantity, as quantregForest inherits it from
  randomForest. `variable_importance` is the share.
- **GBM** (`GbmResult::relative_influence`): the sum over trees of the SSE decrease of each split on the
  pseudo-residuals, which is gbm's `relative.influence`. `feature_importances` is the share.

Before this was fixed, each split added `0.5^depth`: a depth-weighted split count that ignored how much the split
explained.

## Cases and values

| Case | Call | Decisive values |
|---|---|---|
| `tree_importance_rpart` | `rpart(cp = 0, xval = 0)`, 200 rows, with `minsplit`/`minbucket`/`maxdepth` 2/1/3 (random forest builder) and 11/5/4 (quantile forest builder) | 2/1/3: x0 5131.18698490, x1 13.91913754, x2 22.14897477 (7 splits); 11/5/4: x0 5134.37920764, x1 15.41044588, x2 26.02764955 (14 splits); shares at 2/1/3: 0.99301987, 0.00269372, 0.00428641 |
| `gbm_stump_influence` | `gbm(distribution = 'gaussian', n.trees = 50, interaction.depth = 1, n.minobsinnode = 1, shrinkage = 0.1, bag.fraction = 1, train.fraction = 1)` | `relative.influence` 26987.28669483, 43.03372621, 94.25927350; shares 0.99493843, 0.00158652, 0.00347505; `initF` 5.54635808 |
| `rf_importance_purity` | `randomForest(ntree = 500, mtry = 2, nodesize = 5` or `10)`, five seeds | shares at nodesize 5: 0.95322-0.95978, 0.01084-0.01296, 0.02921-0.03534; x0 purity 4940.86-4975.71 |

## Tests

Exact pins at 1e-8 relative (coefficient row, n = 200):

- `test_validate_rf_tree_importance_rpart` (one tree of the forest builder equals rpart);
- `test_validate_qrf_tree_importance_rpart` (the same for the quantile forest);
- `test_validate_gbm_stump_influence` (fitted values, `initF`, split variables, `relative.influence`).

Band tests, on seeded p2a forests:

- `test_validate_rf_importance_band_randomforest`;
- `test_validate_qrf_importance_band_randomforest`.

**Band rule.** The five-seed randomForest range, widened by twice its width on each side, and the ranking
x0 > x2 > x1. A band test outside its band means that p2a's forest and randomForest differ systematically; the
band is never widened. Before the band was set, the p2a forest algorithm was transcribed in R with R's RNG, and it
gave shares 0.95513-0.95883 / 0.01079-0.01199 / 0.03037-0.03299 for the random forest and 0.95964-0.96289 /
0.00938-0.01088 / 0.02760-0.02971 for the quantile forest. The depth-weighted rule on the same forests gave
0.3564 / 0.2865 / 0.3571 and 0.3568 / 0.2842 / 0.3590, with x2 above x0.

## What does not match R

- **Forest importances are not R-pinned (band of five randomForest seeds).** The bootstrap and feature draws use
  p2a's RNG, so a forest cannot be matched draw by draw.
- gbm's Bernoulli and Laplace leaves (ML-94, WT-1) are not pinned; only the Gaussian stump path is.

## Running

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- tree_importance_rpart importance_band gbm_stump_influence
```
