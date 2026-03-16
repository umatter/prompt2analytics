#!/usr/bin/env Rscript
# Mood Two-Sample Test of Scale - R Benchmark
# Compares R implementation performance against p2a Rust
#
# References:
# - Conover (1971), Practical Nonparametric Statistics, pp. 234-235
# - Mielke (1967), Technometrics 9(2):312-314

set.seed(42)

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000)

cat("=== Mood Two-Sample Test of Scale R Benchmarks ===\n\n")

# Try to use microbenchmark if available, otherwise fallback to system.time
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (use_microbenchmark) {
  library(microbenchmark)

  for (n in sizes) {
    x <- runif(n)
    y <- runif(n)

    # Warmup
    invisible(mood.test(x, y))

    # Benchmark
    bm <- microbenchmark(
      mood.test(x, y),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 100 runs)\n", n, med))
  }
} else {
  cat("Note: microbenchmark not available, using system.time fallback\n\n")

  for (n in sizes) {
    x <- runif(n)
    y <- runif(n)

    # Warmup
    invisible(mood.test(x, y))

    # Benchmark with 50 replications
    timing <- system.time(replicate(50, { mood.test(x, y) }))
    med_ms <- timing["elapsed"] * 1000 / 50
    med_us <- med_ms * 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test 1: Equal scales (should give p ≈ 1)
cat("Test 1: Equal scales\n")
x1 <- c(1, 2, 3, 4, 5)
y1 <- c(10, 20, 30, 40, 50)
result1 <- mood.test(x1, y1)
cat(sprintf("  x = c(1,2,3,4,5), y = c(10,20,30,40,50)\n"))
cat(sprintf("  Z = %.4f, p-value = %.4f\n", result1$statistic, result1$p.value))

# Test 2: Different scales
cat("\nTest 2: Different scales\n")
x2 <- c(4.5, 4.8, 5.0, 5.2, 5.5)  # small variance
y2 <- c(1.0, 3.0, 5.0, 7.0, 9.0)  # large variance
result2 <- mood.test(x2, y2)
cat(sprintf("  x (small var) vs y (large var)\n"))
cat(sprintf("  Z = %.4f, p-value = %.4f\n", result2$statistic, result2$p.value))

# Test 3: Data with ties
cat("\nTest 3: Data with ties\n")
x3 <- c(1, 2, 2, 3, 4)
y3 <- c(2, 3, 3, 4, 5)
result3 <- mood.test(x3, y3)
cat(sprintf("  x = c(1,2,2,3,4), y = c(2,3,3,4,5)\n"))
cat(sprintf("  Z = %.4f, p-value = %.4f\n", result3$statistic, result3$p.value))

# Test 4: Larger random samples with different scales
cat("\nTest 4: Larger samples with different scales (n=50 each)\n")
set.seed(123)
x4 <- rnorm(50, mean = 0, sd = 1)   # sd = 1
y4 <- rnorm(50, mean = 0, sd = 3)   # sd = 3
result4 <- mood.test(x4, y4)
cat(sprintf("  x ~ N(0,1), y ~ N(0,3)\n"))
cat(sprintf("  Z = %.4f, p-value = %.4f\n", result4$statistic, result4$p.value))

cat("\nDone.\n")
