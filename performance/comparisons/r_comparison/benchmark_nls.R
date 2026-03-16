#!/usr/bin/env Rscript
# NLS (Nonlinear Least Squares) R Benchmark
# Compares R implementation performance against p2a Rust
#
# Run with: Rscript performance/comparisons/r_comparison/benchmark_nls.R

library(microbenchmark)

set.seed(42)

# ============================================================================
# Data Generation (matching Rust benchmark DGP)
# ============================================================================

generate_exponential_decay_data <- function(n, seed = 42) {
  set.seed(seed)

  # True model: y = 10 * exp(-0.5 * x) + 2 + noise
  x <- seq(0, 5, length.out = n)
  noise <- runif(n, -0.2, 0.2)
  y <- 10 * exp(-0.5 * x) + 2 + noise

  list(x = x, y = y)
}

generate_michaelis_menten_data <- function(n, seed = 42) {
  set.seed(seed)

  # True model: V = 200 * S / (0.1 + S) + noise
  x <- 0.01 * 10^(seq(0, 3, length.out = n))
  noise <- runif(n, -2, 2)
  y <- 200 * x / (0.1 + x) + noise

  list(x = x, y = y)
}

# ============================================================================
# NLS Benchmarks
# ============================================================================

cat("======================================================\n")
cat("R NLS Benchmarks\n")
cat("======================================================\n")

results <- list()

# =============================================================================
# Exponential Decay
# =============================================================================

cat("\n=== Exponential Decay (y = a * exp(-b * x) + c) ===\n")

for (n in c(100, 1000, 10000)) {
  cat(sprintf("  n=%d: ", n))
  data <- generate_exponential_decay_data(n)

  tryCatch({
    bm <- microbenchmark(
      nls(data$y ~ a * exp(-b * data$x) + c,
          start = list(a = 8, b = 0.3, c = 1)),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert to microseconds
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("exp_decay_n", n)]] <- summary(bm)
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# =============================================================================
# Michaelis-Menten
# =============================================================================

cat("\n=== Michaelis-Menten (V = Vmax * S / (Km + S)) ===\n")

for (n in c(100, 1000, 10000)) {
  cat(sprintf("  n=%d: ", n))
  data <- generate_michaelis_menten_data(n)

  tryCatch({
    bm <- microbenchmark(
      nls(data$y ~ Vmax * data$x / (Km + data$x),
          start = list(Vmax = 150, Km = 0.05)),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("mm_n", n)]] <- summary(bm)
  }, error = function(e) {
    cat(sprintf("FAILED: %s\n", e$message))
  })
}

# =============================================================================
# Algorithm Comparison (R uses Gauss-Newton by default)
# =============================================================================

cat("\n=== Algorithm Comparison (n=100) ===\n")

data <- generate_exponential_decay_data(100)

cat("  Gauss-Newton (default): ")
bm_gn <- microbenchmark(
  nls(data$y ~ a * exp(-b * data$x) + c,
      start = list(a = 8, b = 0.3, c = 1),
      algorithm = "default"),
  times = 100,
  unit = "microseconds"
)
cat(sprintf("%.2f us (median)\n", median(bm_gn$time) / 1000))

cat("  Port (bounds support): ")
bm_port <- microbenchmark(
  nls(data$y ~ a * exp(-b * data$x) + c,
      start = list(a = 8, b = 0.3, c = 1),
      algorithm = "port",
      lower = c(0, 0, 0),
      upper = c(100, 10, 10)),
  times = 100,
  unit = "microseconds"
)
cat(sprintf("%.2f us (median)\n", median(bm_port$time) / 1000))

# ============================================================================
# Summary
# ============================================================================

cat("\n======================================================\n")
cat("SUMMARY TABLE (median times in microseconds)\n")
cat("======================================================\n\n")

cat("Exponential Decay:\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("exp_decay_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\nMichaelis-Menten:\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("mm_n", n)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", n, med))
  }
}

cat("\nAlgorithm Comparison (n=100):\n")
cat(sprintf("  Gauss-Newton: %10.2f us\n", median(bm_gn$time) / 1000))
cat(sprintf("  Port:         %10.2f us\n", median(bm_port$time) / 1000))

cat("\n======================================================\n")
cat("Done. Run Rust benchmarks with:\n")
cat("  cargo bench -p p2a-core -- nls\n")
cat("======================================================\n")
