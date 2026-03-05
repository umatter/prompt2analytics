# Conditional Inference Trees (CTree)

## Method

Conditional inference trees use permutation-based statistical tests to select splitting variables, avoiding variable selection bias of traditional CART. Splits on the most significant variable using p-value based criterion.

## R Package Reference

- **Package**: `partykit` / `party`
- **Function**: `ctree(formula, data, control = ctree_control())`
- **CRAN**: https://cran.r-project.org/package=partykit

## R Code for Expected Values

```r
library(partykit)
set.seed(42)
n <- 200
x1 <- runif(n); x2 <- runif(n); x3 <- runif(n) - 0.5
y <- 2 * x1 + 0.5 * x2 + rnorm(n, 0, 0.3)
df <- data.frame(y, x1, x2, x3)

model <- ctree(y ~ x1 + x2 + x3, data = df,
               control = ctree_control(mincriterion = 0.95, minsplit = 20))
pred <- predict(model, df)
cat("R²:", 1 - sum((y - pred)^2) / sum((y - mean(y))^2), "\n")
varimp(model)
```

## Rust Test Reference

- **File**: `crates/p2a-core/tests/validation_ml_advanced.rs`
- **Tests**: `test_validate_ctree_regression`, `test_validate_ctree_classification`, `test_validate_ctree_predict`

## Tolerance

- Regression R²: > 0.50 (typically > 0.90)
- Classification accuracy: > 0.75
- Root p-values: x1 significant (< 0.05), x3 non-significant
