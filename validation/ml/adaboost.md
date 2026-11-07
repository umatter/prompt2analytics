# AdaBoost (M1, SAMME, R2)

## Method

`p2a_core::ml::adaboost` boosts CART weak learners of depth `max_depth` on case weights, one weighted
grower for every depth and mode:

- **AdaBoost.M1** (Freund & Schapire 1997): labels -1/+1, `alpha = lr * 0.5 * ln((1 - e) / e)`, stops when
  `e >= 0.5`; predictions are the signed scores.
- **SAMME** (Zhu, Zou, Rosset & Hastie 2009, Algorithm 2): integer class codes, K >= 2,
  `alpha = lr * (ln((1 - e) / e) + ln(K - 1))`, weights `w * exp(alpha * miss)`, stops when `e >= 1 - 1/K`;
  predictions are the argmax label of the summed votes (ties: the smallest label).
- **AdaBoost.R2** (Drucker 1997): least-squares weak learners, weighted median of the rounds.

A first learner no better than chance is an error; a later one ends the fit with a `warnings` note naming
the round and the error. `max_depth` is limited to 1-30 (rpart's `maxdepth` limit). The default stump is
rpart's weighted-Gini stump, not a weighted-error stump: default M1 fits differ from versions before this
learner.

The grower sorts each column once per fit, then a node owns a segment of every column's order array and a
split stably partitions it, which costs O(p n) per tree level.

## R reference

`validation/r/cases/ml_boosting_t3.R`, case `ml_adaboost_samme` (data `generated/adaboost_sim.csv`, n = 120).
The rounds are written out in R because `adabag` is not a case dependency; the weak learner is the installed
`rpart::rpart(weights = w, control = rpart.control(maxdepth = d, minsplit = 2, minbucket = 1, cp = 0, xval = 0,
maxcompete = 0, maxsurrogate = 0))`, method `class` or `anova`. The SAMME loop follows
`adabag::boosting(coeflearn = "Zhu", boos = FALSE)`.

## Decisive values

| Quantity | R value |
|---|---|
| SAMME depth 1, round 1 weighted error | 47/120 = 0.391666... |
| SAMME depth 1, round 1 alpha | ln(146/47) = 1.133459019998278 |
| SAMME depth 1, weights after round 1 | 73 rows 1/219, 47 rows 2/141 |
| SAMME depth 1, training accuracy after 8 rounds | 0.666666... |
| SAMME depth 2, round 1 alpha | 1.7047480922384253 |
| M1 depth 1, round 1 alpha | 0.50580045583924 |
| M1 depth 3, round 1 alpha | 0.7196083812871166 |

## Tests

In `crates/p2a-core/src/ml/adaboost.rs`, registered as `ml_adaboost`:
`test_validate_adaboost_samme_depth1`, `test_validate_adaboost_samme_depth2`,
`test_validate_adaboost_samme_learning_rate`, `test_validate_adaboost_samme_first_round_weights`,
`test_validate_adaboost_m1_depth1`, `test_validate_adaboost_m1_depth3`,
`test_validate_adaboost_r2_first_learner`.

Properties, not compared with R: `test_samme_with_two_classes_is_m1` (K = 2 doubles M1's alpha),
`test_weak_tree_partition_matches_naive_regrowth` (the partitioned grower equals a re-sorting reference bit
for bit, depths 1-5, both targets), `test_adaboost_r2_max_depth_changes_the_fit`.

## Tolerances

`Tol::coef(120)` (1e-8 relative for n = 120) for weights, errors and scores; labels and accuracies exact.

## Not pinned

The R2 loop beyond its first learner: Drucker's loop deviations are the open finding ML-110.
