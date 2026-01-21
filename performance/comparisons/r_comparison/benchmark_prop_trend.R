#!/usr/bin/env Rscript
# prop.trend.test R Benchmark

library(microbenchmark)

set.seed(42)

cat("=== prop.trend.test R Benchmarks ===\n")

# Different number of groups
group_sizes <- c(3, 5, 10, 20)

for (k in group_sizes) {
  # Generate data with trend
  n <- rep(100, k)
  p <- seq(0.2, 0.8, length.out = k)
  x <- rbinom(k, n, p)

  bm <- microbenchmark(
    prop.trend.test(x, n),
    times = 1000,
    unit = "microseconds"
  )

  cat(sprintf("  k=%d groups: %.2f us (median)\n", k, median(bm$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
# Example from R documentation
smokers <- c(83, 90, 129, 70)
patients <- c(86, 93, 136, 82)
result <- prop.trend.test(smokers, patients)
cat(sprintf("X-squared: %.6f\n", result$statistic))
cat(sprintf("df: %d\n", result$parameter))
cat(sprintf("p-value: %.10f\n", result$p.value))
