#!/usr/bin/env Rscript
# STL R Benchmark

library(microbenchmark)

set.seed(42)

cat("=== STL R Benchmarks ===\n")

sizes <- c(120, 240, 480, 960)

for (n in sizes) {
  # Generate seasonal time series
  t <- 1:n
  trend <- 100 + 0.5 * t
  seasonal <- 10 * sin(2 * pi * t / 12)
  noise <- rnorm(n, sd = 2)
  y <- ts(trend + seasonal + noise, frequency = 12)

  bm <- microbenchmark(
    stl(y, s.window = "periodic"),
    times = 50,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d: %.2f us (median)\n", n, median(bm$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
n <- 120
t <- 1:n
trend <- 100 + 0.5 * t
seasonal <- 10 * sin(2 * pi * t / 12)
y <- ts(trend + seasonal, frequency = 12)

result <- stl(y, s.window = "periodic")
cat(sprintf("Trend range: [%.2f, %.2f]\n", min(result$time.series[,2]), max(result$time.series[,2])))
cat(sprintf("Seasonal range: [%.2f, %.2f]\n", min(result$time.series[,1]), max(result$time.series[,1])))
