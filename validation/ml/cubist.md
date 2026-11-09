# Cubist Rule-Based Regression

## Method

Cubist is a rule-based regression model that builds regression trees and extracts rules with linear models in the terminal nodes. It supports committee models (boosting-like ensembles) and instance-based (k-NN) corrections.

## R Package Reference

- **Package**: `Cubist` (v0.4.2+)
- **Function**: `cubist(x, y, committees = 1, neighbors = 0)`
- **CRAN**: https://cran.r-project.org/package=Cubist

## Illustrative R code (not run: C50 and Cubist are not installed on the validation machine)

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
- **Tests**: `test_cubist_regression_properties`, `test_cubist_committees_properties`, `test_cubist_predict_properties`

## Leaf model with collinear attributes (R-pinned)

Registry entry `cubist_leaf_model` in `validation/methods/numeric_hygiene.json`; case `inv_cubist_leaf_model`
(`validation/r/cases/numeric_cubist.R`). A leaf's linear model is solved by least squares that detects the
rank: an attribute collinear with earlier ones is eliminated (coefficient 0), as the Cubist C code does,
where a generalized inverse split the coefficient between the duplicated columns (0.7714 each instead of
1.5428 on `x1`). Pinned to `lm(y ~ x1 + x1dup)`: intercept 1.7324117064449327, `x1` 1.542842534411597,
`x1dup` NA (0 in p2a). Test: `test_validate_cubist_leaf_model_duplicated_column`.

## Property thresholds (no R parity)

The tests check DGP recovery only, and no number is compared with Cubist output.

- Training R²: > 0.70 (typically > 0.90)
- Out-of-sample R²: > 0.50
