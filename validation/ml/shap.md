# SHAP (SHapley Additive exPlanations)

## Method

SHAP values provide a unified measure of feature importance based on Shapley values from cooperative game theory. For tree ensembles, TreeSHAP computes exact Shapley values in polynomial time. The implementation supports both tree-based and kernel-based SHAP.

## R Package Reference

- **Package**: `SHAP` / `treeshap` / `iml`
- **Functions**: `treeshap()`, `kernelshap()`
- **CRAN**: https://cran.r-project.org/package=treeshap

## R Code for Expected Values

```r
library(randomForest)
library(treeshap)
set.seed(42)
n <- 100
x1 <- runif(n); x2 <- runif(n); x3 <- runif(n) - 0.5
y <- 2 * x1 + 0.5 * x2 + rnorm(n, 0, 0.3)
df <- data.frame(x1, x2, x3)

rf <- randomForest(y ~ ., data = cbind(df, y), ntree = 20, maxnodes = 32)
unified <- randomForest.unify(rf, df)
shap <- treeshap(unified, df)

# Mean absolute SHAP values
colMeans(abs(shap$shaps))
# Expected: x1 > x2 > x3

# Additivity check
base_value <- mean(y)
pred <- predict(rf, df)
shap_sum <- rowSums(shap$shaps)
max(abs(pred - (base_value + shap_sum)))  # Should be ~0
```

## Rust Test Reference

- **File**: `crates/p2a-core/tests/validation_ml_advanced.rs`
- **Tests**: `test_validate_shap_tree_ensemble`, `test_validate_shap_feature_importance_ordering`

## Tolerance

- Feature importance ordering: x1 > x2 > x3
- Mean SHAP additivity error: < 1.0 (approximate tree SHAP)
- SHAP value dimensions: n_samples x n_features
