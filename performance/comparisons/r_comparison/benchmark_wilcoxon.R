#!/usr/bin/env Rscript
# Wilcoxon Test R Benchmark
# Compares R implementation performance against p2a Rust
#
# Run with: Rscript performance/comparisons/r_comparison/benchmark_wilcoxon.R

library(microbenchmark)

set.seed(42)

# ============================================================================
# Data Generation
# ============================================================================

generate_two_sample_data <- function(n1, n2, seed = 42) {
  set.seed(seed)
  x <- rnorm(n1, mean = 5, sd = 2)
  y <- rnorm(n2, mean = 6, sd = 2)
  list(x = x, y = y)
}

generate_paired_data <- function(n, seed = 42) {
  set.seed(seed)
  x <- rnorm(n, mean = 100, sd = 15)
  y <- x + rnorm(n, mean = -5, sd = 10)  # Paired with shift
  list(x = x, y = y)
}

generate_one_sample_data <- function(n, seed = 42) {
  set.seed(seed)
  x <- rnorm(n, mean = 10, sd = 3)
  list(x = x, mu = 9.5)
}

# ============================================================================
# Wilcoxon Rank Sum (Mann-Whitney U) Benchmarks
# ============================================================================

cat("======================================================\n")
cat("R Wilcoxon Test Benchmarks\n")
cat("======================================================\n")

results <- list()

cat("\n=== Wilcoxon Rank Sum (Mann-Whitney U) ===\n")
cat("Two independent samples, normal approximation\n\n")

for (n in c(100, 1000, 10000)) {
  cat(sprintf("  n=%d (each group): ", n))
  data <- generate_two_sample_data(n, n)

  tryCatch({
    bm <- microbenchmark(
      wilcox.test(data$x, data$y, exact = FALSE, correct = TRUE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert to microseconds
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("rank_sum_n", n)]] <- summary(bm)
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# ============================================================================
# Wilcoxon Signed Rank Benchmarks
# ============================================================================

cat("\n=== Wilcoxon Signed Rank (Paired) ===\n")
cat("Paired samples, normal approximation\n\n")

for (n in c(100, 1000, 10000)) {
  cat(sprintf("  n=%d pairs: ", n))
  data <- generate_paired_data(n)

  tryCatch({
    bm <- microbenchmark(
      wilcox.test(data$x, data$y, paired = TRUE, exact = FALSE, correct = TRUE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("signed_rank_n", n)]] <- summary(bm)
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# ============================================================================
# One-Sample Signed Rank Benchmarks
# ============================================================================

cat("\n=== Wilcoxon Signed Rank (One-Sample) ===\n")
cat("One sample vs hypothesized median\n\n")

for (n in c(100, 1000, 10000)) {
  cat(sprintf("  n=%d: ", n))
  data <- generate_one_sample_data(n)

  tryCatch({
    bm <- microbenchmark(
      wilcox.test(data$x, mu = data$mu, exact = FALSE, correct = TRUE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("one_sample_n", n)]] <- summary(bm)
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# ============================================================================
# Exact Test Benchmarks (Small Samples)
# ============================================================================

cat("\n=== Exact Test (Small Samples, No Ties) ===\n")

for (n in c(5, 10, 15, 20)) {
  cat(sprintf("  n=%d (each group): ", n))
  data <- generate_two_sample_data(n, n)

  tryCatch({
    bm <- microbenchmark(
      wilcox.test(data$x, data$y, exact = TRUE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("exact_n", n)]] <- summary(bm)
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

cat("Rank Sum (two independent samples):\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("rank_sum_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\nSigned Rank (paired samples):\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("signed_rank_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\nOne-Sample Signed Rank:\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("one_sample_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\nExact Test (small samples):\n")
for (n in c(5, 10, 15, 20)) {
  key <- paste0("exact_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\n======================================================\n")
cat("Done. Run Rust benchmarks with:\n")
cat("  cargo bench -p p2a-core -- wilcoxon\n")
cat("======================================================\n")
