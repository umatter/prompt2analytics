#!/usr/bin/env Rscript
# McNemar's Chi-Squared Test R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)
if (use_microbenchmark) {
  library(microbenchmark)
}

set.seed(42)

cat("=== McNemar's Test R Benchmarks ===\n\n")

# McNemar's test operates on summary counts (b, c), not raw data
# So we benchmark the number of test invocations

# Test data (from R documentation)
Performance <- matrix(c(794, 86, 150, 570), nrow = 2)

# Extract b and c for repeated calls
b <- Performance[1, 2]  # 150
c <- Performance[2, 1]  # 86

cat("Test configuration:\n")
cat(sprintf("  b = %d, c = %d\n", b, c))
cat(sprintf("  Using microbenchmark: %s\n\n", use_microbenchmark))

# Benchmark different numbers of test invocations
n_tests <- c(100, 1000, 10000)

cat("Performance Results:\n")
cat("-------------------\n")

for (n in n_tests) {
  if (use_microbenchmark) {
    # Use microbenchmark for accurate timing
    bm <- microbenchmark(
      for (i in 1:n) mcnemar.test(Performance),
      times = 10,
      unit = "microseconds"
    )
    med_time <- median(bm$time) / 1000  # Convert to microseconds
    cat(sprintf("  %d test(s): %.2f µs (median of 10 runs)\n", n, med_time))
  } else {
    # Fallback using system.time
    times <- replicate(10, {
      start <- Sys.time()
      for (i in 1:n) mcnemar.test(Performance)
      as.numeric(Sys.time() - start) * 1e6  # Convert to microseconds
    })
    med_time <- median(times)
    cat(sprintf("  %d test(s): %.2f µs (median of 10 runs)\n", n, med_time))
  }
}

cat("\n")

# Also test with correction = FALSE
cat("Without continuity correction:\n")
cat("------------------------------\n")

for (n in n_tests) {
  if (use_microbenchmark) {
    bm <- microbenchmark(
      for (i in 1:n) mcnemar.test(Performance, correct = FALSE),
      times = 10,
      unit = "microseconds"
    )
    med_time <- median(bm$time) / 1000
    cat(sprintf("  %d test(s): %.2f µs (median of 10 runs)\n", n, med_time))
  } else {
    times <- replicate(10, {
      start <- Sys.time()
      for (i in 1:n) mcnemar.test(Performance, correct = FALSE)
      as.numeric(Sys.time() - start) * 1e6
    })
    med_time <- median(times)
    cat(sprintf("  %d test(s): %.2f µs (median of 10 runs)\n", n, med_time))
  }
}

cat("\n=== Benchmark Complete ===\n")
