#!/usr/bin/env Rscript
# Classical Decomposition R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Classical Decomposition R Benchmarks ===\n\n")

# Generate seasonal time series
generate_seasonal <- function(n, period = 12) {
  t <- 1:n
  trend <- 100 + 0.5 * t
  seasonal <- 10 * sin(2 * pi * t / period)
  noise <- rnorm(n, sd = 1)
  trend + seasonal + noise
}

# Test configurations
configs <- c(100, 1000, 10000)

cat("Additive Decomposition Benchmarks\n")
cat("---------------------------------\n\n")

for (n in configs) {
  x <- ts(generate_seasonal(n), frequency = 12)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      decompose(x, type = "additive"),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n", med))
    cat(sprintf("  Mean:   %.2f µs\n", mean(bm$time) / 1000))
    cat(sprintf("  Min:    %.2f µs\n", min(bm$time) / 1000))
    cat(sprintf("  Max:    %.2f µs\n\n", max(bm$time) / 1000))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      decompose(x, type = "additive")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\nMultiplicative Decomposition Benchmarks\n")
cat("---------------------------------------\n\n")

# Generate multiplicative data (all positive)
generate_mult <- function(n, period = 12) {
  t <- 1:n
  trend <- 100 + 0.5 * t
  seasonal <- 1 + 0.2 * sin(2 * pi * t / period)
  noise <- 1 + rnorm(n, sd = 0.01)
  trend * seasonal * noise
}

for (n in configs) {
  x <- ts(generate_mult(n), frequency = 12)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      decompose(x, type = "multiplicative"),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n\n", med))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      decompose(x, type = "multiplicative")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Generate test series
x <- ts(generate_seasonal(48), frequency = 12)

cat("Additive decomposition (n=48, period=12):\n")
result_add <- decompose(x, type = "additive")
cat("\nSeasonal figure:\n")
print(result_add$figure)

cat("\n\nMultiplicative decomposition:\n")
x_mult <- ts(generate_mult(48), frequency = 12)
result_mult <- decompose(x_mult, type = "multiplicative")
cat("\nSeasonal figure:\n")
print(result_mult$figure)

cat("\n\nTrend (first 10 values):\n")
print(head(result_add$trend, 10))

cat("\n\nSeasonal (first 10 values):\n")
print(head(result_add$seasonal, 10))

cat("\n\nRandom (first 10 values):\n")
print(head(result_add$random, 10))
