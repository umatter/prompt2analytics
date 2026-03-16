#!/usr/bin/env Rscript
# AR Model Fitting R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== AR Model Fitting R Benchmarks ===\n\n")

# Generate AR(2) time series of different lengths
generate_ar2 <- function(n) {
  x <- numeric(n)
  x[1:2] <- rnorm(2)
  for (t in 3:n) {
    x[t] <- 0.7 * x[t-1] - 0.2 * x[t-2] + rnorm(1, sd = 0.5)
  }
  x
}

# Test configurations
configs <- c(100, 1000, 10000)

cat("Yule-Walker Method Benchmarks\n")
cat("------------------------------\n\n")

for (n in configs) {
  x <- generate_ar2(n)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      ar(x, method = "yule-walker"),
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
      ar(x, method = "yule-walker")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\nBurg Method Benchmarks\n")
cat("-----------------------\n\n")

for (n in configs) {
  x <- generate_ar2(n)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      ar(x, method = "burg"),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n\n", med))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      ar(x, method = "burg")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\nOLS Method Benchmarks\n")
cat("----------------------\n\n")

for (n in configs) {
  x <- generate_ar2(n)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      ar(x, method = "ols"),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n\n", med))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      ar(x, method = "ols")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Generate test series
x <- generate_ar2(100)

cat("AR(2) process simulation: phi = (0.7, -0.2)\n\n")

cat("Yule-Walker method:\n")
result_yw <- ar(x, method = "yule-walker")
print(result_yw)

cat("\n\nBurg method:\n")
result_burg <- ar(x, method = "burg")
print(result_burg)

cat("\n\nOLS method:\n")
result_ols <- ar(x, method = "ols")
print(result_ols)

cat("\n\nFixed order AR(2):\n")
result_fixed <- ar(x, aic = FALSE, order.max = 2)
print(result_fixed)

cat("\n\nAIC values (relative):\n")
print(result_yw$aic)
