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
  `test_cubist_rule_model_is_the_composite`, `test_cubist_committee_target`,
  `test_cubist_smoothing_coefficient_is_validated` (smoothing and committees);
  `test_cubist_usage_counts_cases`, `test_cubist_importance_is_unit_free`,
  `test_cubist_usage_averages_committees`, `test_cubist_display_without_usage` (attribute usage);
  `test_cubist_instance_correction_closed_form`, `test_cubist_neighbours_are_scale_free`,
  `test_cubist_training_metrics_leave_the_row_out`,
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
`k = 0` keeps the leaf model; a large `k` approaches the root model. `smoothing_coefficient` must be
finite and non-negative (otherwise `InvalidSpecification`).

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

`release`, p2a-core with the `ci` features, against `f80b0131` (v0.5.0); quiet re-time, 2026-11-15 (this tree at
`2d0bb8b6`): one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; base and this tree interleaved per row
with the order rotated per round, each process the median of its calls after a warm-up (calls calibrated to about 0.8
s per process); cells are the median of the process medians over five rounds (nine or ten for rows within 1.05-1.15x);
1-minute load average 1.0-3.0. "ratio" is this tree over the base, "min ratio" the ratio of the minima.
Friedman-1 data, p = 5, noise sd 1. No R timing (Cubist is not installed).

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| fit, defaults | 100 | 0.216 | 0.246 | 1.14 | 1.14 |
|  | 1,000 | 4.66 | 4.73 | 1.01 | 1.01 |
|  | 10,000 | 47.2 | 46.4 | 0.98 | 0.99 |
| fit, `committees = 5` | 100 | 1.39 | 1.64 | 1.18 | 1.19 |
|  | 1,000 | 23.8 | 23.7 | 0.99 | 0.99 |
|  | 10,000 | 245 | 236 | 0.96 | 0.96 |
| fit, `neighbors = 5` (leave-one-out training pass) | 100 | 0.804 | 0.519 | 0.65 | 0.64 |
|  | 1,000 | 67.7 | 27.3 | 0.40 | 0.41 |
|  | 10,000 | 6,909 | 2,228 | 0.32 | 0.32 |
| `cubist_predict`, 1,000 new rows, `neighbors = 5` | 100 | 5.58 | 2.57 | 0.46 | 0.46 |
|  | 1,000 | 62.8 | 21.7 | 0.34 | 0.34 |
|  | 10,000 | 698 | 213 | 0.30 | 0.31 |

The neighbour rows are faster because the k nearest cases are selected in linear time instead of sorting every
training case per prediction. Peak RSS of one `neighbors = 5` fit at n = 10,000 (`/usr/bin/time -v`, whole process):
6.5 MB base, 6.8 MB this tree on the quiet re-time (median of three; 7.1 MB, 1.11x, on the loaded run; a small total: the result now keeps the n model predictions of the
training rows and the p column ranges for the neighbour step, and the leave-one-out pass holds a distance buffer
of n entries). Cubist itself changed by two lines between v0.5.0 and the base of this change, but the OLS solve
its node models call (`normal_equations_inverse`) changed in between too, so these rows include that change as well
as Cubist's own. The fit rows above 1.1x by the minimum on the loaded run and the peak RSS row were signed off on 2026-11-14 (rule 5);
on the quiet re-time those rows are at 0.96-1.01x, and the n = 100 fits (1.14x defaults,
1.18x with five committees) are above 1.1x and go to the user.
