#!/usr/bin/env Rscript
# ACF/PACF/CCF R Benchmark
# Compares R stats::acf performance against p2a Rust implementation
#
# Usage: Rscript benchmark_acf.R

# Check for microbenchmark
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (has_microbenchmark) {
  library(microbenchmark)
}

set.seed(42)

# Generate AR(1) time series (matching Rust benchmark DGP)
generate_ar1 <- function(n, phi = 0.7, seed = 42) {
  set.seed(seed)
  x <- numeric(n)
  x[1] <- runif(1, -1, 1)
  for (t in 2:n) {
    x[t] <- phi * x[t-1] + runif(1, -0.5, 0.5)
  }
  x
}

cat("=== ACF/PACF/CCF R Benchmarks ===\n\n")

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000)

benchmark_function <- function(name, fn, sizes, n_reps = 100) {
  cat(sprintf("--- %s ---\n", name))

  for (n in sizes) {
    x <- generate_ar1(n, 0.7, 42)
    y <- generate_ar1(n, 0.7, 123)  # For CCF

    if (has_microbenchmark) {
      if (name == "CCF") {
        bm <- microbenchmark(
          fn(x, y, plot = FALSE),
          times = n_reps,
          unit = "microseconds"
        )
      } else {
        bm <- microbenchmark(
          fn(x, plot = FALSE),
          times = n_reps,
          unit = "microseconds"
        )
      }
      med <- median(bm$time) / 1000  # Convert to microseconds
      cat(sprintf("  n=%d: %.2f us (median of %d)\n", n, med, n_reps))
    } else {
      # Fallback without microbenchmark
      n_reps_fallback <- 50
      if (name == "CCF") {
        timing <- system.time(replicate(n_reps_fallback, { fn(x, y, plot = FALSE) }))
      } else {
        timing <- system.time(replicate(n_reps_fallback, { fn(x, plot = FALSE) }))
      }
      ms_per_call <- timing["elapsed"] * 1000 / n_reps_fallback
      us_per_call <- ms_per_call * 1000
      cat(sprintf("  n=%d: %.2f us (mean of %d, system.time)\n", n, us_per_call, n_reps_fallback))
    }
  }
  cat("\n")
}

# Run benchmarks
benchmark_function("ACF", acf, sizes)
benchmark_function("PACF", pacf, sizes)
benchmark_function("CCF", ccf, sizes)

cat("=== Benchmark Complete ===\n")
