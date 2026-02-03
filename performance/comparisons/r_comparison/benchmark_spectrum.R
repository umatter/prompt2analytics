#!/usr/bin/env Rscript
# Spectral Density Estimation R Benchmark
# Compares R stats::spectrum performance against p2a Rust implementation

# Check for microbenchmark availability
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

set.seed(42)

# =============================================================================
# Data Generation Function
# =============================================================================

# Generate time series with two frequency components + noise
# (Same DGP as Rust benchmark)
generate_spectrum_data <- function(n) {
  t <- 0:(n - 1)
  signal1 <- sin(2 * pi * 0.1 * t)
  signal2 <- 0.5 * sin(2 * pi * 0.25 * t)
  noise <- runif(n, -0.3, 0.3)
  signal1 + signal2 + noise
}

# =============================================================================
# Benchmark Functions
# =============================================================================

cat("=== Spectral Density Estimation R Benchmarks ===\n\n")

sizes <- c(100, 1000, 10000, 100000)

if (has_microbenchmark) {
  library(microbenchmark)

  # ==========================================================================
  # Raw Periodogram Benchmark
  # ==========================================================================

  cat("--- Raw Periodogram (spans=NULL, taper=0) ---\n")
  for (n in sizes) {
    x <- generate_spectrum_data(n)

    bm <- microbenchmark(
      spec.pgram(x, spans = NULL, taper = 0, detrend = FALSE, demean = TRUE, plot = FALSE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert ns to µs
    cat(sprintf("  n=%6d: %12.2f µs (median)\n", n, med))
  }

  # ==========================================================================
  # Smoothed Periodogram Benchmark
  # ==========================================================================

  cat("\n--- Smoothed Periodogram (spans=c(3,3), taper=0.1) ---\n")
  for (n in sizes) {
    x <- generate_spectrum_data(n)

    bm <- microbenchmark(
      spec.pgram(x, spans = c(3, 3), taper = 0.1, detrend = TRUE, demean = TRUE, plot = FALSE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert ns to µs
    cat(sprintf("  n=%6d: %12.2f µs (median)\n", n, med))
  }

  # ==========================================================================
  # AR-based Spectrum Benchmark
  # ==========================================================================

  cat("\n--- AR-based Spectrum (spec.ar) ---\n")
  ar_sizes <- c(100, 1000, 10000)  # Skip 100000 as spec.ar is slower
  for (n in ar_sizes) {
    x <- generate_spectrum_data(n)

    bm <- microbenchmark(
      spec.ar(x, plot = FALSE),
      times = 50,  # Fewer iterations for AR (slower)
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert ns to µs
    cat(sprintf("  n=%6d: %12.2f µs (median)\n", n, med))
  }

} else {
  # Fallback without microbenchmark
  cat("Note: microbenchmark not available, using system.time()\n\n")

  # ==========================================================================
  # Raw Periodogram Benchmark (fallback)
  # ==========================================================================

  cat("--- Raw Periodogram (spans=NULL, taper=0) ---\n")
  for (n in sizes) {
    x <- generate_spectrum_data(n)

    timing <- system.time(replicate(100, {
      spec.pgram(x, spans = NULL, taper = 0, detrend = FALSE, demean = TRUE, plot = FALSE)
    }))

    avg_ms <- timing["elapsed"] * 1000 / 100
    cat(sprintf("  n=%6d: %12.2f µs (mean of 100)\n", n, avg_ms * 1000))
  }

  # ==========================================================================
  # Smoothed Periodogram Benchmark (fallback)
  # ==========================================================================

  cat("\n--- Smoothed Periodogram (spans=c(3,3), taper=0.1) ---\n")
  for (n in sizes) {
    x <- generate_spectrum_data(n)

    timing <- system.time(replicate(100, {
      spec.pgram(x, spans = c(3, 3), taper = 0.1, detrend = TRUE, demean = TRUE, plot = FALSE)
    }))

    avg_ms <- timing["elapsed"] * 1000 / 100
    cat(sprintf("  n=%6d: %12.2f µs (mean of 100)\n", n, avg_ms * 1000))
  }

  # ==========================================================================
  # AR-based Spectrum Benchmark (fallback)
  # ==========================================================================

  cat("\n--- AR-based Spectrum (spec.ar) ---\n")
  ar_sizes <- c(100, 1000, 10000)
  for (n in ar_sizes) {
    x <- generate_spectrum_data(n)

    timing <- system.time(replicate(50, {
      spec.ar(x, plot = FALSE)
    }))

    avg_ms <- timing["elapsed"] * 1000 / 50
    cat(sprintf("  n=%6d: %12.2f µs (mean of 50)\n", n, avg_ms * 1000))
  }
}

cat("\n=== Benchmark Complete ===\n")
