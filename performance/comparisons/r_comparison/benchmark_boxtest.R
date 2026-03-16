#!/usr/bin/env Rscript
# Box-Pierce and Ljung-Box Test R Benchmark
# Compares R implementation performance against p2a Rust

suppressPackageStartupMessages({
  if (requireNamespace("microbenchmark", quietly = TRUE)) {
    library(microbenchmark)
    use_microbenchmark <- TRUE
  } else {
    use_microbenchmark <- FALSE
    cat("Note: microbenchmark not installed, using system.time() fallback\n")
  }
})

set.seed(42)

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000)

cat("=== Box.test R Benchmarks ===\n")
cat("Testing Ljung-Box and Box-Pierce with lag=10\n\n")

for (n in sizes) {
  # Generate white noise data
  x <- rnorm(n)

  if (use_microbenchmark) {
    # Ljung-Box benchmark
    bm_lb <- microbenchmark(
      Box.test(x, lag = 10, type = "Ljung-Box"),
      times = 100,
      unit = "microseconds"
    )
    med_lb <- median(bm_lb$time) / 1000  # ns to us

    # Box-Pierce benchmark
    bm_bp <- microbenchmark(
      Box.test(x, lag = 10, type = "Box-Pierce"),
      times = 100,
      unit = "microseconds"
    )
    med_bp <- median(bm_bp$time) / 1000  # ns to us

    cat(sprintf("n=%d:\n", n))
    cat(sprintf("  Ljung-Box:  %.2f µs (median)\n", med_lb))
    cat(sprintf("  Box-Pierce: %.2f µs (median)\n", med_bp))
  } else {
    # Fallback using system.time
    reps <- 100

    time_lb <- system.time(replicate(reps, { Box.test(x, lag = 10, type = "Ljung-Box") }))
    time_bp <- system.time(replicate(reps, { Box.test(x, lag = 10, type = "Box-Pierce") }))

    med_lb <- time_lb["elapsed"] * 1000000 / reps  # seconds to µs
    med_bp <- time_bp["elapsed"] * 1000000 / reps

    cat(sprintf("n=%d:\n", n))
    cat(sprintf("  Ljung-Box:  %.2f µs (avg of %d)\n", med_lb, reps))
    cat(sprintf("  Box-Pierce: %.2f µs (avg of %d)\n", med_bp, reps))
  }
}

cat("\n=== Benchmark Complete ===\n")
