#!/usr/bin/env Rscript
# Kolmogorov-Smirnov Test R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
has_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Kolmogorov-Smirnov Test R Benchmarks ===\n\n")

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000)

cat("Two-Sample KS Test Benchmarks:\n")
cat("-------------------------------\n")

for (n in sizes) {
  # Generate two samples from slightly different distributions
  x <- rnorm(n, mean = 0, sd = 1)
  y <- rnorm(n, mean = 0.1, sd = 1)  # Slightly shifted

  if (has_microbenchmark) {
    bm <- microbenchmark(
      ks.test(x, y),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert from nanoseconds to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 100)\n", n, med))
  } else {
    # Fallback without microbenchmark
    timing <- system.time(replicate(50, { ks.test(x, y) }))
    avg_us <- timing["elapsed"] * 1000000 / 50
    cat(sprintf("  n=%d: %.2f us (average of 50)\n", n, avg_us))
  }
}

cat("\nOne-Sample KS Test (vs Normal) Benchmarks:\n")
cat("-------------------------------------------\n")

for (n in sizes) {
  x <- rnorm(n)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      ks.test(x, "pnorm"),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 100)\n", n, med))
  } else {
    timing <- system.time(replicate(50, { ks.test(x, "pnorm") }))
    avg_us <- timing["elapsed"] * 1000000 / 50
    cat(sprintf("  n=%d: %.2f us (average of 50)\n", n, avg_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test Case 1: Two-sample shifted
x <- c(1.2, 1.5, 1.8, 2.1, 2.4)
y <- c(2.0, 2.5, 3.0, 3.5, 4.0)
result <- ks.test(x, y)
cat("Test 1 - Two-sample shifted:\n")
cat(sprintf("  D = %.6f, p-value = %.6f\n\n", result$statistic, result$p.value))

# Test Case 2: Identical samples
x <- 1:5
y <- 1:5
result <- suppressWarnings(ks.test(x, y))
cat("Test 2 - Identical samples:\n")
cat(sprintf("  D = %.6f, p-value = %.6f\n\n", result$statistic, result$p.value))

# Test Case 3: One-sample normal
x <- c(-0.56, 0.12, -0.89, 0.45, 0.23, -0.11, 0.78, -0.34,
       0.56, -0.67, 0.89, -0.23, 0.01, 0.45, -0.78, 0.34,
       -0.45, 0.67, -0.12, 0.23)
result <- ks.test(x, "pnorm")
cat("Test 3 - One-sample vs N(0,1):\n")
cat(sprintf("  D = %.6f, p-value = %.6f\n\n", result$statistic, result$p.value))

# Test Case 4: One-sample uniform
x <- seq(1, 10) / 11
result <- ks.test(x, "punif")
cat("Test 4 - One-sample vs U(0,1):\n")
cat(sprintf("  D = %.6f, p-value = %.6f\n\n", result$statistic, result$p.value))

cat("=== Done ===\n")
