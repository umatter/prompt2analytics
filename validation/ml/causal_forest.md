# Causal Forest Validation

Reference: grf 2.x `causal_forest` (not installed; formula pins and property tests).

## Cases

| Case | Dataset | What it holds |
|---|---|---|
| `cf_aipw_formula` | `validation/datasets/generated/cf_aipw_formula.csv` | grf's AIPW scores and `average_treatment_effect` written out |
| `cf_little_bags` | `validation/datasets/generated/cf_little_bags.csv` | pooled leaf prediction, little-bags variance, objective-Bayes debiaser |
| `cf_split_node` | `validation/datasets/generated/cf_split_node.csv` | relabeling, causal and regression split on one node |
| `cf_confounded` | `validation/datasets/generated/cf_confounded.csv` | confounded design (ML-20) |
| `cf_rct` | `validation/datasets/generated/cf_rct.csv` | randomised design (ML-21) |
| `cf_rare` | `validation/datasets/generated/cf_rare.csv` | rare treatment (ML-74) |

The R code is `validation/r/cases/ml_causal_forest.R`.

## Values

## Tests
