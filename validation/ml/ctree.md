# Conditional Inference Trees (CTree)

## Method

Conditional inference trees use permutation-based statistical tests to select splitting variables, avoiding variable selection bias of traditional CART. Splits on the most significant variable using p-value based criterion.

## R Package Reference

- **Package**: `partykit` / `party`
- **Function**: `ctree(formula, data, control = ctree_control())`
- **CRAN**: https://cran.r-project.org/package=partykit

## Illustrative R code (not run; no number is compared with partykit)

partykit is installed on the validation machine, but p2a's node statistics are known to differ from it (ML-26,
ML-27), so a pin would record those differences. party and coin are not installed.

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

## What is pinned

- The split point against rpart: `test_validate_ctree_split_matches_rpart` (`crates/p2a-core/src/ml/ctree.rs`).
- Parameter validation: `test_ctree_rejects_minsplit_below_twice_minbucket`, `test_ctree_minbucket_zero_acts_as_one`,
  `test_ctree_options_parse_aliases_and_reject_unknown`, `test_ctree_maximum_teststat_is_not_the_quadratic_one`.
  `minsplit` below `2 * minbucket` is refused, because no node of that size can be split into two children of
  `minbucket` rows. `teststat = maximum` with a classification target is refused until ML-26 and ML-27 fix the
  statistic. The `MonteCarlo` and `Teststatistic` test types are not implemented and are refused.

- The root's variable-selection statistic and its p-values below 1e-16 (registry entry
  `ctree_root_statistics` in `validation/methods/numeric_hygiene.json`; case `tail_ctree_root`,
  `validation/r/cases/numeric_ctree.R`). partykit is not installed, so the case writes out the quadratic rank
  statistic, `(n - 1) cor(rank(x), y)^2`, and `pchisq(lower.tail = FALSE)`. Two of the three root p-values are
  below 1e-16 (1.73e-19 and 9.92e-47), where `1 - cdf` returned 0 for both and the choice between them fell to
  column order; the tail is now computed directly, so the variable with the larger statistic is chosen.
  Test: `test_validate_ctree_root_tail_tie_break`.

## What is not R parity

- The node statistics and p-values (ML-26, ML-27). No statistic or p-value is compared with partykit output.
- The importance definition: `Sum(1 - p)` over the splits on a feature, not partykit's `varimp`.

## DGP property tests

`test_ctree_regression_properties`, `test_ctree_classification_properties` and `test_ctree_predict_properties`
(`crates/p2a-core/tests/validation_ml_advanced.rs`) check DGP recovery by thresholds, despite their names: they are
property tests, not R parity, and they will be renamed (X-10).

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
