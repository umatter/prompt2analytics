#!/usr/bin/env Rscript
# Friedman Rank Sum Test - R Benchmark
# Compares R implementation performance against p2a Rust
#
# References:
# - Friedman (1937), JASA 32(200):675-701
# - Hollander & Wolfe (1973), Nonparametric Statistical Methods, pp. 139-146

set.seed(42)

# Benchmark at different numbers of blocks (rows)
# Each block has 3 treatments
n_treatments <- 3
sizes <- c(30, 100, 300, 1000)

cat("=== Friedman Rank Sum Test R Benchmarks ===\n\n")

# Try to use microbenchmark if available, otherwise fallback to system.time
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (use_microbenchmark) {
  library(microbenchmark)

  for (n_blocks in sizes) {
    # Generate matrix: n_blocks rows x n_treatments columns
    data <- matrix(
      runif(n_blocks * n_treatments) * 10 + rep(0:(n_treatments-1), each=n_blocks) * 2,
      nrow = n_blocks,
      ncol = n_treatments,
      byrow = FALSE
    )

    # Warmup
    invisible(friedman.test(data))

    # Benchmark
    bm <- microbenchmark(
      friedman.test(data),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  n=%d blocks: %.2f us (median of 100 runs)\n", n_blocks, med))
  }
} else {
  cat("Note: microbenchmark not available, using system.time fallback\n\n")

  for (n_blocks in sizes) {
    data <- matrix(
      runif(n_blocks * n_treatments) * 10 + rep(0:(n_treatments-1), each=n_blocks) * 2,
      nrow = n_blocks,
      ncol = n_treatments,
      byrow = FALSE
    )

    # Warmup
    invisible(friedman.test(data))

    # Benchmark with 50 replications
    timing <- system.time(replicate(50, { friedman.test(data) }))
    med_ms <- timing["elapsed"] * 1000 / 50
    med_us <- med_ms * 1000
    cat(sprintf("  n=%d blocks: %.2f us (median of 50 runs)\n", n_blocks, med_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test 1: RoundingTimes example from R documentation
cat("Test 1: RoundingTimes from R docs\n")
RoundingTimes <- matrix(c(5.40, 5.50, 5.55,
                          5.85, 5.70, 5.75,
                          5.20, 5.60, 5.50,
                          5.55, 5.50, 5.40,
                          5.90, 5.85, 5.70,
                          5.45, 5.55, 5.60,
                          5.40, 5.40, 5.35,
                          5.45, 5.50, 5.35,
                          5.25, 5.15, 5.00,
                          5.85, 5.80, 5.70,
                          5.25, 5.20, 5.10,
                          5.65, 5.55, 5.45),
                        nrow=12, byrow=TRUE)
result1 <- friedman.test(RoundingTimes)
cat(sprintf("  Q = %.4f, df = %d, p-value = %.6f\n", result1$statistic, result1$parameter, result1$p.value))

# Test 2: Simple 3x3 perfect ordering
cat("\nTest 2: Perfect ordering 3x3\n")
data2 <- matrix(c(1, 2, 3,
                  1, 2, 3,
                  1, 2, 3),
                nrow = 3, byrow = TRUE)
result2 <- friedman.test(data2)
cat(sprintf("  Q = %.4f, df = %d, p-value = %.6f\n", result2$statistic, result2$parameter, result2$p.value))

# Test 3: Data with ties
cat("\nTest 3: Data with ties\n")
data3 <- matrix(c(1, 1, 2,
                  2, 2, 3,
                  1, 2, 2),
                nrow = 3, byrow = TRUE)
result3 <- friedman.test(data3)
cat(sprintf("  Q = %.4f, df = %d, p-value = %.6f\n", result3$statistic, result3$parameter, result3$p.value))

cat("\nDone.\n")
