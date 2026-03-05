# Cubist Rule-Based Regression

## Method

Cubist is a rule-based regression model that builds regression trees and extracts rules with linear models in the terminal nodes. It supports committee models (boosting-like ensembles) and instance-based (k-NN) corrections.

## R Package Reference

- **Package**: `Cubist` (v0.4.2+)
- **Function**: `cubist(x, y, committees = 1, neighbors = 0)`
- **CRAN**: https://cran.r-project.org/package=Cubist

## R Code for Expected Values

```r
library(Cubist)
set.seed(42)
n <- 200
x1 <- runif(n); x2 <- runif(n); x3 <- runif(n) - 0.5
y <- 2 * x1 + 0.5 * x2 + rnorm(n, 0, 0.3)
df <- data.frame(x1, x2, x3)

model <- cubist(df, y, committees = 1)
pred <- predict(model, df)
cat("Train R²:", 1 - sum((y - pred)^2) / sum((y - mean(y))^2), "\n")

model5 <- cubist(df, y, committees = 5)
pred5 <- predict(model5, df)
cat("5-committee R²:", 1 - sum((y - pred5)^2) / sum((y - mean(y))^2), "\n")
```

## Rust Test Reference

- **File**: `crates/p2a-core/tests/validation_ml_advanced.rs`
- **Tests**: `test_validate_cubist_regression`, `test_validate_cubist_committees`, `test_validate_cubist_predict`

## Tolerance

- Training R²: > 0.70 (typically > 0.90)
- Out-of-sample R²: > 0.50
