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

## Target types and missing values

- **Target.** `run_ctree` (and `ml_ctree`) takes a numeric target (integer and decimal
  columns): it fits a classification tree when the target has at most 10 distinct values
  and fewer than n / 2, rounded down, otherwise a regression tree (unchanged; an override is
  ML-87, Tier 4). A text, categorical or boolean target becomes the class codes 0..K-1 in
  R `factor()` order, reported in `CtreeResult::class_names` (JSON `class_names`) and a
  "Class labels:" line of the Display. A text target that the rule would fit as a
  regression (more than 10 classes, or not fewer classes than n / 2, rounded down) is refused
  with an explicit error instead of fitting a regression on the codes.
- **Missing values.** A missing value (null or NaN) in the target or a feature is an
  error naming the column (Fail; no surrogate splits). `±Inf` is refused as well.
- **Tests.** `test_na_ctree_class_labels`, `test_na_ml_wrappers_int64_target_equals_float64`
  and `test_na_ml_wrappers_refuse_missing_values` (`crates/p2a-core/src/ml/extract.rs`),
  `test_ctree_classifies_rule` (`crates/p2a-core/src/ml/ctree.rs`),
  `test_na_mcp_tree_tools_integer_and_missing_targets` and
  `test_mcp_c50_ctree_text_targets` (`crates/p2a-mcp/src/tools/handlers/ml.rs`). They
  compare p2a with p2a and pin no R number.
