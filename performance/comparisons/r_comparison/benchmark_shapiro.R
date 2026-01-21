#!/usr/bin/env Rscript
# Shapiro-Wilk Test R Benchmark
# Compares R implementation performance against p2a Rust
#
# Run with: Rscript performance/comparisons/r_comparison/benchmark_shapiro.R

cat("======================================================\n")
cat("R Shapiro-Wilk Test Benchmarks\n")
cat("======================================================\n\n")

# Check for microbenchmark package
if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  cat("Note: microbenchmark package not available, using system.time\n\n")
  use_microbenchmark <- FALSE
} else {
  library(microbenchmark)
  use_microbenchmark <- TRUE
}

set.seed(42)

# ============================================================================
# Data Generation
# ============================================================================

generate_normal_data <- function(n, seed = 42) {
  set.seed(seed)
  rnorm(n)
}

generate_mixed_data <- function(n, seed = 42) {
  set.seed(seed)
  c(rnorm(n * 0.8), rexp(n * 0.2))
}

# ============================================================================
# Benchmarks
# ============================================================================

results <- list()

cat("=== Shapiro-Wilk Test (Normal Data) ===\n\n")

for (n in c(10, 50, 100, 500, 1000, 2000, 5000)) {
  cat(sprintf("  n=%d: ", n))
  data <- generate_normal_data(n)

  tryCatch({
    if (use_microbenchmark) {
      bm <- microbenchmark(
        shapiro.test(data),
        times = 100,
        unit = "microseconds"
      )
      med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
      cat(sprintf("%.2f us (median)\n", med))
      results[[paste0("normal_n", n)]] <- summary(bm)
    } else {
      timing <- system.time(replicate(100, shapiro.test(data)))
      avg_ms <- timing["elapsed"] * 1000 / 100
      cat(sprintf("%.2f ms (avg)\n", avg_ms))
    }
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

cat("\n=== Shapiro-Wilk Test (Mixed Data) ===\n\n")

for (n in c(10, 50, 100, 500, 1000)) {
  cat(sprintf("  n=%d: ", n))
  data <- generate_mixed_data(n)

  tryCatch({
    if (use_microbenchmark) {
      bm <- microbenchmark(
        shapiro.test(data),
        times = 100,
        unit = "microseconds"
      )
      med <- median(bm$time) / 1000
      cat(sprintf("%.2f us (median)\n", med))
      results[[paste0("mixed_n", n)]] <- summary(bm)
    } else {
      timing <- system.time(replicate(100, shapiro.test(data)))
      avg_ms <- timing["elapsed"] * 1000 / 100
      cat(sprintf("%.2f ms (avg)\n", avg_ms))
    }
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# ============================================================================
# Summary
# ============================================================================

cat("\n======================================================\n")
cat("SUMMARY TABLE (median times in microseconds)\n")
cat("======================================================\n\n")

if (use_microbenchmark) {
  cat("Normal data:\n")
  for (n in c(10, 50, 100, 500, 1000, 2000, 5000)) {
    key <- paste0("normal_n", n)
    if (!is.null(results[[key]])) {
      med <- results[[key]]$median
      cat(sprintf("  n=%5d: %10.2f us\n", n, med))
    }
  }

  cat("\nMixed data:\n")
  for (n in c(10, 50, 100, 500, 1000)) {
    key <- paste0("mixed_n", n)
    if (!is.null(results[[key]])) {
      med <- results[[key]]$median
      cat(sprintf("  n=%5d: %10.2f us\n", n, med))
    }
  }
}

cat("\n======================================================\n")
cat("Done. Run Rust benchmarks with:\n")
cat("  cargo bench -p p2a-core -- shapiro\n")
cat("======================================================\n")
