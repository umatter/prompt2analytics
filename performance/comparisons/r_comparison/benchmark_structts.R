#!/usr/bin/env Rscript
# StructTS R Benchmark
# Compares R StructTS() performance against p2a Rust implementation

# Try to load microbenchmark, fall back to system.time if not available
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)
if (has_microbenchmark) {
  library(microbenchmark)
}

set.seed(42)

# Generate local level series (random walk + observation noise)
generate_local_level <- function(n) {
  level <- 100
  y <- numeric(n)
  for (i in 1:n) {
    level <- level + rnorm(1, 0, 1)  # level noise
    y[i] <- level + rnorm(1, 0, 2.5)  # observation noise
  }
  y
}

# Generate BSM series (level + trend + seasonality)
generate_bsm <- function(n, period = 12) {
  level <- 100
  slope <- 0.5
  y <- numeric(n)
  for (t in 1:n) {
    level <- level + slope + rnorm(1, 0, 0.25)
    slope <- slope + rnorm(1, 0, 0.025)
    seasonal <- 10 * sin(2 * pi * t / period)
    y[t] <- level + seasonal + rnorm(1, 0, 1.5)
  }
  y
}

cat("=== StructTS R Benchmarks ===\n\n")

# Local Level Model
cat("1. Local Level Model (type = 'level')\n")
for (n in c(100, 1000, 10000)) {
  y <- generate_local_level(n)
  y_ts <- ts(y)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      StructTS(y_ts, type = "level"),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert ns to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(20, { StructTS(y_ts, type = "level") }))
    cat(sprintf("  n=%d: %.2f ms (median of 20 runs)\n", n, timing["elapsed"] * 1000 / 20))
  }
}

cat("\n")

# Local Linear Trend Model
cat("2. Local Linear Trend Model (type = 'trend')\n")
for (n in c(100, 1000, 10000)) {
  y <- generate_local_level(n)
  y_ts <- ts(y)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      StructTS(y_ts, type = "trend"),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(20, { StructTS(y_ts, type = "trend") }))
    cat(sprintf("  n=%d: %.2f ms (median of 20 runs)\n", n, timing["elapsed"] * 1000 / 20))
  }
}

cat("\n")

# Basic Structural Model (BSM)
cat("3. Basic Structural Model (type = 'BSM')\n")
for (n in c(100, 1000, 10000)) {
  y <- generate_bsm(n, 12)
  y_ts <- ts(y, frequency = 12)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      StructTS(y_ts, type = "BSM"),
      times = 20,  # Fewer runs for BSM as it's slower
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d (period=12): %.2f us (median of 20 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(10, { StructTS(y_ts, type = "BSM") }))
    cat(sprintf("  n=%d (period=12): %.2f ms (median of 10 runs)\n", n, timing["elapsed"] * 1000 / 10))
  }
}

cat("\n")

# KalmanFilter/KalmanRun using makeARIMA + KalmanRun
cat("4. Kalman Filter (using makeARIMA + KalmanRun)\n")
cat("   Note: R's KalmanRun is accessed via arima internals\n")
for (n in c(100, 1000, 10000)) {
  y <- generate_local_level(n)
  y_ts <- ts(y)

  # Use StructTS as proxy since it calls Kalman internally
  if (has_microbenchmark) {
    bm <- microbenchmark(
      StructTS(y_ts, type = "level"),
      times = 20,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 20 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(20, { StructTS(y_ts, type = "level") }))
    cat(sprintf("  n=%d: %.2f ms (median of 20 runs)\n", n, timing["elapsed"] * 1000 / 20))
  }
}

cat("\n=== End of StructTS Benchmarks ===\n")
