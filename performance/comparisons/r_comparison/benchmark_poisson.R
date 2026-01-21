#!/usr/bin/env Rscript
# Exact Poisson Test R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Exact Poisson Test R Benchmarks ===\n\n")

# Test configurations for one-sample tests
one_sample_configs <- list(
  list(x = 10, t = 1),
  list(x = 100, t = 10),
  list(x = 1000, t = 100),
  list(x = 10000, t = 1000)
)

cat("One-Sample Poisson Test Benchmarks\n")
cat("-----------------------------------\n\n")

for (config in one_sample_configs) {
  cat(sprintf("x=%d, t=%d:\n", config$x, config$t))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      poisson.test(config$x, config$t),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  Median: %.2f µs\n", med))
    cat(sprintf("  Mean:   %.2f µs\n", mean(bm$time) / 1000))
    cat(sprintf("  Min:    %.2f µs\n", min(bm$time) / 1000))
    cat(sprintf("  Max:    %.2f µs\n\n", max(bm$time) / 1000))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      poisson.test(config$x, config$t)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

# Test configurations for two-sample tests
two_sample_configs <- list(
  list(x1 = 5, x2 = 10, t1 = 1, t2 = 2),
  list(x1 = 50, x2 = 100, t1 = 10, t2 = 20),
  list(x1 = 500, x2 = 1000, t1 = 100, t2 = 200),
  list(x1 = 5000, x2 = 10000, t1 = 1000, t2 = 2000)
)

cat("\nTwo-Sample Poisson Test Benchmarks\n")
cat("-----------------------------------\n\n")

for (config in two_sample_configs) {
  cat(sprintf("x=[%d,%d], t=[%d,%d]:\n", config$x1, config$x2, config$t1, config$t2))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      poisson.test(c(config$x1, config$x2), c(config$t1, config$t2)),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n", med))
    cat(sprintf("  Mean:   %.2f µs\n", mean(bm$time) / 1000))
    cat(sprintf("  Min:    %.2f µs\n", min(bm$time) / 1000))
    cat(sprintf("  Max:    %.2f µs\n\n", max(bm$time) / 1000))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      poisson.test(c(config$x1, config$x2), c(config$t1, config$t2))
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# One-sample test validation
cat("One-sample test: poisson.test(137, 24.19893)\n")
result1 <- poisson.test(137, 24.19893)
print(result1)

# Two-sample test validation
cat("\n\nTwo-sample test: poisson.test(c(11, 21), c(800, 3011))\n")
result2 <- poisson.test(c(11, 21), c(800, 3011))
print(result2)

# Test with different alternatives
cat("\n\nOne-sided tests:\n")
cat("\nGreater alternative: poisson.test(15, 10, alternative='greater')\n")
print(poisson.test(15, 10, alternative='greater'))

cat("\nLess alternative: poisson.test(5, 10, alternative='less')\n")
print(poisson.test(5, 10, alternative='less'))
