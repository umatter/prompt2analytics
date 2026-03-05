# Model-Based Boosting (MBoost)

## Method

MBoost implements component-wise gradient boosting with selectable base learners (linear or tree stumps). At each iteration, it fits each base learner to the negative gradient and selects the best-fitting one. Provides built-in variable selection through the boosting process.

## R Package Reference

- **Package**: `mboost` (v2.9+)
- **Functions**: `glmboost()` (linear), `gamboost()` (smooth), `blackboost()` (tree)
- **CRAN**: https://cran.r-project.org/package=mboost

## R Code for Expected Values

```r
library(mboost)
set.seed(42)
n <- 200
x1 <- runif(n); x2 <- runif(n); x3 <- runif(n) - 0.5
y <- 2 * x1 + 0.5 * x2 + rnorm(n, 0, 0.3)
df <- data.frame(y, x1, x2, x3)

# Linear base learner
model_lin <- glmboost(y ~ x1 + x2 + x3, data = df,
                       control = boost_control(mstop = 200, nu = 0.1))
coef(model_lin)
pred <- predict(model_lin, df)
cat("Linear R²:", 1 - sum((y - pred)^2) / sum((y - mean(y))^2), "\n")

# Tree base learner
model_tree <- blackboost(y ~ x1 + x2 + x3, data = df,
                          control = boost_control(mstop = 100, nu = 0.1),
                          tree_controls = ctree_control(maxdepth = 3))
pred_tree <- predict(model_tree, df)
cat("Tree R²:", 1 - sum((y - pred_tree)^2) / sum((y - mean(y))^2), "\n")
```

## Rust Test Reference

- **File**: `crates/p2a-core/tests/validation_ml_advanced.rs`
- **Tests**: `test_validate_mboost_linear_regression`, `test_validate_mboost_tree_regression`, `test_validate_mboost_predict`, `test_validate_mboost_poisson`

## Tolerance

- Linear R²: > 0.70 (typically > 0.90)
- Tree R²: > 0.75
- Out-of-sample R²: > 0.50
- Coefficients: x1 ~ 2.0, x2 ~ 0.5 (within 0.5 of true values)
