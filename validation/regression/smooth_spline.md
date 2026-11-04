# Smoothing spline (`regression::smooth_spline`)

**Method:** cubic smoothing spline with the smoothing parameter chosen by GCV (default), ordinary
leave-one-out CV, a target df, spar or lambda.

**R reference:** `stats::smooth.spline(cv = FALSE | TRUE, df = 6)`, case `smoothers_gcv_periodic_loess`
(`validation/r/cases/regression_smoothers_t3.R`).

**Tests:** not R-pinned (REG-29, REG-30); the property tests are listed below.
