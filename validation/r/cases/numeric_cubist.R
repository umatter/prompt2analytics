# validation/r/cases/numeric_cubist.R -- (rank-deficient solves): lm with a duplicated column, as a Cubist leaf model
# fit sees it (the later column is NA and contributes 0).

case("inv_cubist_leaf_model", "ml/cubist", "lm with a duplicated column (Cubist leaf model)", function(ctx) {
  set.seed(20261028)
  n <- 30
  x1 <- round(runif(n, 0, 10), 6)
  y <- round(2 + 1.5 * x1 + rnorm(n, 0, 0.7), 6)
  d <- ctx$gen_dataset("inv_cubist_leaf", data.frame(y = y, x1 = x1, x1dup = x1))
  fit <- lm(y ~ x1 + x1dup, data = d)
  stopifnot(is.na(coef(fit)["x1dup"]))
  X <- cbind(1, d$x1, d$x1dup); bg <- drop(MASS::ginv(crossprod(X)) %*% crossprod(X, d$y))
  list(values = list(coef = named(coef(fit)), coef_ginv = vec(bg)),
       reference = "stats::lm (x1dup NA)", call = "lm(y ~ x1 + x1dup)", packages = c("MASS"))
})
