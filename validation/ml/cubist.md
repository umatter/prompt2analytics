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

- `crates/p2a-core/tests/validation_ml_advanced.rs`: `test_cubist_regression_properties`,
  `test_cubist_committees_properties`, `test_cubist_predict_properties` (property thresholds below)
- `crates/p2a-core/src/ml/cubist.rs`: `test_cubist_min_improvement_stops_splits`,
  `test_cubist_split_rule_is_ssd_share`, `test_cubist_min_improvement_is_a_share` (split rule);
  `test_cubist_smoothing_reaches_the_root`, `test_cubist_smoothing_coefficient_direction`,
  `test_cubist_rule_model_is_the_composite`, `test_cubist_committee_target` (smoothing and committees);
  `test_cubist_usage_counts_cases`, `test_cubist_importance_is_unit_free`,
  `test_cubist_usage_averages_committees` (attribute usage); `test_cubist_instance_correction_closed_form`,
  `test_cubist_neighbours_are_scale_free`, `test_cubist_training_metrics_leave_the_row_out`,
  `test_cubist_neighbors_limit` (instance correction).

## Splits, smoothing and committees

A node is split when the best split removes more than `min_improvement` (default 0.01, in [0, 1)) of
the node's sum of squared deviations of the target, subject to `min_split`, `min_bucket` and `max_depth`.

Every node carries a least-squares model. A rule's model is its leaf model smoothed with every ancestor's,
from the leaf up to the root (Quinlan 1992):

    p <- (n p + k q) / (n + k)

per ancestor, with `p` the model so far, `q` the ancestor's model, `n` the training cases of the child
on the path and `k = smoothing_coefficient` (default 15). Each step is linear in the models, so the
composite is one linear model, stored per rule: prediction evaluates it, and the rules print it.
`k = 0` keeps the leaf model; a large `k` approaches the root model.

With `committees > 1`, member `m + 1` is fitted to `2y - pred_m`, `pred_m` the prediction of member
`m`, as R's Cubist ("if the first model over-predicted by 2, the next model's target is y - 2"); the
prediction is the members' mean.

## Attribute usage and importance

`usage_conditions[j]` is the percent of training cases whose rule tests attribute `j` in its
conditions, `usage_model[j]` the percent whose rule model has a non-zero coefficient on `j`, each
averaged over the committees, as Cubist's `usage`. `variable_importance` is
`(usage_conditions + usage_model) / 2` (caret's `varImp` for Cubist), normalised to sum to one. Both
are unit-free: rescaling a feature leaves them unchanged.

## Instance correction (`neighbors`)

`neighbors` is 0 to 9, as in R. With `k > 0` a prediction at `x` is

    model(x) + mean over the k nearest training cases of (y_i - model(x_i))

(Quinlan 1993), `model` the committee-average rule model, distances Euclidean on features divided by
their training range (a constant feature contributes nothing), neighbours ordered by distance and then
row index. On the training data a row is left out of its own neighbour set, so `train_rmse` and
`train_r_squared` do not see the row's own target; `cubist_predict` uses every training case.

## Known differences from Cubist

- Node models use the attributes tested on the path to the node (all attributes at the root), with no
  term elimination; Cubist and M5 choose other attribute sets and simplify the models. Because every
  rule model contains the root model, `usage_model` is 100 for every non-aliased attribute on most fits,
  and the importance ranking then comes from `usage_conditions`.
- Rules are not simplified or pruned after the tree is grown.
- The instance correction weights the k neighbours equally; Cubist weights them by distance.

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

## Wall-clock

`release`, p2a-core with the `ci` features, against `f80b0131` (v0.5.0), five rounds with the sides interleaved per
row (order alternating per round), one core (`taskset -c 2`), one rayon thread, 2026-11-14. **Preliminary**: the load
average was 33-70 during the runs (other builds on the machine), so ratios within about 0.05 of 1 are noise. Cells are
the median of the round medians (each the median of 1-21 calls after a warm-up); "ratio" is this tree over the base,
"min ratio" the ratio of the minima. Friedman-1 data, p = 5, noise sd 1. No R timing (Cubist is not installed).

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| fit, defaults | 100 | 0.47 | 0.48 | 1.03 | 0.99 |
|  | 1,000 | 7.29 | 7.51 | 1.03 | 1.03 |
|  | 10,000 | 71.8 | 72.5 | 1.01 | 1.20 |
| fit, `committees = 5` | 100 | 2.50 | 2.54 | 1.02 | 1.03 |
|  | 1,000 | 37.0 | 39.8 | 1.08 | 1.02 |
|  | 10,000 | 380 | 387 | 1.02 | 0.99 |
| fit, `neighbors = 5` (leave-one-out training pass) | 100 | 1.43 | 1.02 | 0.71 | 0.73 |
|  | 1,000 | 111 | 52.3 | 0.47 | 0.46 |
|  | 10,000 | 11,425 | 4,278 | 0.37 | 0.38 |
| `cubist_predict`, 1,000 new rows, `neighbors = 5` | 100 | 8.93 | 4.83 | 0.54 | 0.54 |
|  | 1,000 | 101 | 41.3 | 0.41 | 0.40 |
|  | 10,000 | 1,134 | 397 | 0.35 | 0.36 |

The neighbour rows are faster because the k nearest cases are selected in linear time instead of sorting every
training case per prediction. Peak RSS of one `neighbors = 5` fit at n = 10,000 (`/usr/bin/time -v`, whole process):
6.4 MB base, 7.1 MB this tree (1.11x of a small total: the result now keeps the n model predictions of the
training rows and the p column ranges for the neighbour step, and the leave-one-out pass holds a distance buffer
of n entries). Cubist changed by two lines between v0.5.0 and the lane base, so these rows are this
package's.
