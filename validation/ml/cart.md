# CART (`ml::cart`)

## Method

`cart()` and `run_cart()` fit a regression (`anova`) or classification (`gini`, `information`) tree as
`rpart` 4.1.23 does for continuous predictors, with unit weights, the default priors, 0/1 loss and no
surrogate or competitor splits.

## R reference

`rpart::rpart(xval = <fold vector>, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)` and `rpart::prune`.
Case file: `validation/r/cases/ml_cart_cp.R`. Registry: `validation/methods/ml_cart_cp.json`.

**Status: R-pinned** (see the registry for the quantities).
