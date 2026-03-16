#!/usr/bin/env Rscript
# GLS R Benchmark

library(microbenchmark)
library(nlme)  # For gls()

set.seed(42)

cat("=== GLS R Benchmarks ===\n")

# Cap at 1000: nlme::gls with corAR1 creates n×n correlation matrix, O(n³) at n=10000
sizes <- c(100, 1000)

for (n in sizes) {
  # Generate AR(1) correlated errors
  rho <- 0.7
  e <- numeric(n)
  e[1] <- rnorm(1)
  for (i in 2:n) {
    e[i] <- rho * e[i-1] + rnorm(1)
  }

  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 1 + 2*x1 + 3*x2 + e

  df <- data.frame(y = y, x1 = x1, x2 = x2, time = 1:n)

  bm <- microbenchmark(
    gls(y ~ x1 + x2, data = df, correlation = corAR1(form = ~ time)),
    times = 20,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d: %.2f us (median)\n", n, median(bm$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
n <- 100
rho <- 0.7
e <- numeric(n)
e[1] <- rnorm(1)
for (i in 2:n) {
  e[i] <- rho * e[i-1] + rnorm(1, sd = sqrt(1 - rho^2))
}
x <- rnorm(n)
y <- 1 + 2*x + e
df <- data.frame(y = y, x = x, time = 1:n)

result <- gls(y ~ x, data = df, correlation = corAR1(form = ~ time))
cat(sprintf("Intercept: %.6f\n", coef(result)[1]))
cat(sprintf("Slope: %.6f\n", coef(result)[2]))
cat(sprintf("Estimated rho: %.6f\n", coef(result$modelStruct$corStruct, unconstrained = FALSE)))
