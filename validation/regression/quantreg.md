# Validation: Quantile regression (rq, summary.rq)

## R code

`validation/r/cases/reg_quantreg_br.R` writes three cases (R 4.3.3, quantreg 6.1):

- `rq_br_se`: `rq(y ~ x, tau, method = "br")` on T1's `validation/datasets/generated/quantreg_hetero.csv`
  (100 rows, columns `y`, `x`) at tau 0.02, 0.25, 0.5 and 0.75, with `summary.rq(se = "iid" | "nid" | "ker",
  covariance = TRUE)`, the bandwidths, `rq.fit.fnb` on the dispersion-standardised outcome, y x 1e-6 and 1e6,
  the nonunique location models and the n = 5 subsample;
- `rq_br_mtcars`: `rq(mpg ~ wt + hp + qsec)`, `rq(mpg ~ wt + hp - 1)` and `rq(mpg ~ 1)` on T1's
  `validation/datasets/generated/mtcars.csv` (n = 32) with the same SE types;
- `rq_boot_xy`: `boot.rq(cbind(1, x), y, tau, R = 50, U = U)` (the xy-pair bootstrap of `summary.rq(se =
  "boot")`) on a committed resample index matrix, `validation/datasets/generated/rq_boot_indices.csv` (100 rows x 50
  integer columns `b01`-`b50`, 1-based row indices; sha256
  `5449244ce3a51ab04ab384afe658922a5cde7311d22fc0ff25249cf0d811ffa7`).

Rebuild with `Rscript validation/r/run.R --write rq_br_se rq_br_mtcars rq_boot_xy`.
