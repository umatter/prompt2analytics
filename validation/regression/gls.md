# Generalized least squares (`regression::gls`)

**Method:** GLS with identity, AR(1) or compound-symmetry errors, optionally within groups, with the
correlation parameter fixed or estimated by REML or ML.

**R reference:** `nlme::gls(correlation = corCompSymm(form = ~ 1 | g) | corAR1(form = ~ 1 | g),
method = "REML" | "ML")`, case `gls_compsymm` (`validation/r/cases/regression_smoothers_t3.R`); case
`gls_ar1` (`validation/r/cases/regression_legacy.R`).

**Tests:** see the `gls_estimated_correlation` entry of `validation/methods/regression_smoothers_t3.json`.
