#!/usr/bin/env Rscript
# smooth.spline R Benchmark

library(microbenchmark)

set.seed(42)

cat("=== smooth.spline R Benchmarks ===\n")

sizes <- c(100, 1000, 10000)

for (n in sizes) {
  x <- seq(0, 4*pi, length.out = n)
  y <- sin(x) + rnorm(n, sd = 0.3)

  # With specified df
  bm_df <- microbenchmark(
    smooth.spline(x, y, df = 10),
    times = 50,
    unit = "microseconds"
  )

  # With cross-validation
  bm_cv <- microbenchmark(
    smooth.spline(x, y),
    times = 50,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d:\n", n))
  cat(sprintf("    df=10: %.2f us (median)\n", median(bm_df$time) / 1000))
  cat(sprintf("    cv:    %.2f us (median)\n", median(bm_cv$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
x <- 1:20
y <- sin(x/5)
result <- smooth.spline(x, y, df = 6)
cat(sprintf("df: %.6f\n", result$df))
cat(sprintf("lambda: %.10f\n", result$lambda))
cat(sprintf("spar: %.6f\n", result$spar))
