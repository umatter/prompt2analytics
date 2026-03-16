#!/usr/bin/env Rscript
# Holt-Winters R Benchmark
# Compares R HoltWinters() performance against p2a Rust implementation

# Try to load microbenchmark, fall back to system.time if not available
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)
if (has_microbenchmark) {
  library(microbenchmark)
}

set.seed(42)

# Generate synthetic time series with multiplicative seasonality
generate_seasonal_data <- function(n, period) {
  seasonal_pattern <- 0.8 + 0.4 * sin(seq(0, 2*pi, length.out = period + 1)[-(period + 1)])

  y <- numeric(n)
  for (t in 1:n) {
    trend <- 100 + 0.5 * t
    seasonal <- seasonal_pattern[((t - 1) %% period) + 1]
    noise <- 1 + runif(1) * 0.05 - 0.025
    y[t] <- trend * seasonal * noise
  }
  return(y)
}

cat("=== Holt-Winters R Benchmarks ===\n\n")

# Test different dataset sizes with period=12
sizes <- c(100, 1000, 10000)
period <- 12

cat("1. Multiplicative seasonality (parameter optimization)\n")
cat("   Period:", period, "\n\n")

for (n in sizes) {
  y <- generate_seasonal_data(n, period)
  ts_y <- ts(y, frequency = period)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      HoltWinters(ts_y, seasonal = "multiplicative"),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert from nanoseconds to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(50, { HoltWinters(ts_y, seasonal = "multiplicative") }))
    avg_ms <- timing["elapsed"] * 1000 / 50
    cat(sprintf("  n=%d: %.2f ms (avg of 50 runs)\n", n, avg_ms))
  }
}

cat("\n2. Fixed parameters (no optimization)\n\n")

for (n in sizes) {
  y <- generate_seasonal_data(n, period)
  ts_y <- ts(y, frequency = period)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      HoltWinters(ts_y, alpha = 0.2, beta = 0.1, gamma = 0.3, seasonal = "multiplicative"),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(50, {
      HoltWinters(ts_y, alpha = 0.2, beta = 0.1, gamma = 0.3, seasonal = "multiplicative")
    }))
    avg_ms <- timing["elapsed"] * 1000 / 50
    cat(sprintf("  n=%d: %.2f ms (avg of 50 runs)\n", n, avg_ms))
  }
}

cat("\n3. Different seasonal periods (n = 5 * period)\n\n")

periods <- c(4, 12, 24, 52)

for (p in periods) {
  n <- p * 5  # 5 full cycles
  y <- generate_seasonal_data(n, p)
  ts_y <- ts(y, frequency = p)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      HoltWinters(ts_y, seasonal = "multiplicative"),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  period=%d (n=%d): %.2f us (median of 50 runs)\n", p, n, med))
  } else {
    timing <- system.time(replicate(50, { HoltWinters(ts_y, seasonal = "multiplicative") }))
    avg_ms <- timing["elapsed"] * 1000 / 50
    cat(sprintf("  period=%d (n=%d): %.2f ms (avg of 50 runs)\n", p, n, avg_ms))
  }
}

cat("\n4. Additive vs Multiplicative comparison (n=240)\n\n")

n <- 240
y <- generate_seasonal_data(n, period)
ts_y <- ts(y, frequency = period)

if (has_microbenchmark) {
  bm_add <- microbenchmark(
    HoltWinters(ts_y, seasonal = "additive"),
    times = 50
  )
  bm_mult <- microbenchmark(
    HoltWinters(ts_y, seasonal = "multiplicative"),
    times = 50
  )
  cat(sprintf("  Additive: %.2f us (median)\n", median(bm_add$time) / 1000))
  cat(sprintf("  Multiplicative: %.2f us (median)\n", median(bm_mult$time) / 1000))
} else {
  timing_add <- system.time(replicate(50, { HoltWinters(ts_y, seasonal = "additive") }))
  timing_mult <- system.time(replicate(50, { HoltWinters(ts_y, seasonal = "multiplicative") }))
  cat(sprintf("  Additive: %.2f ms (avg)\n", timing_add["elapsed"] * 1000 / 50))
  cat(sprintf("  Multiplicative: %.2f ms (avg)\n", timing_mult["elapsed"] * 1000 / 50))
}

cat("\n=== Benchmark Complete ===\n")
