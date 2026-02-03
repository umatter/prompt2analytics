#!/usr/bin/env Rscript
# Canonical Correlation Analysis R Benchmark
# Compares R cancor() performance against p2a Rust implementation

# Check if microbenchmark is available
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (has_microbenchmark) {
  library(microbenchmark)
}

set.seed(42)

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000, 100000)

cat("=== Canonical Correlation Analysis R Benchmarks ===\n\n")

# Fixed number of variables: 5 X variables, 3 Y variables
p <- 5
q <- 3

for (n in sizes) {
  # Generate random data
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  Y <- matrix(rnorm(n * q), nrow = n, ncol = q)

  # Add some correlation between X and Y
  common <- rnorm(n)
  X[, 1] <- X[, 1] + common
  Y[, 1] <- Y[, 1] + 0.8 * common

  if (has_microbenchmark) {
    bm <- microbenchmark(
      cancor(X, Y),
      times = if (n <= 10000) 100 else 20,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("n=%d: %.2f us (median of %d runs)\n", n, med, length(bm$time)))
  } else {
    # Fallback without microbenchmark
    n_reps <- if (n <= 10000) 50 else 10
    times <- numeric(n_reps)

    for (i in 1:n_reps) {
      start <- Sys.time()
      result <- cancor(X, Y)
      end <- Sys.time()
      times[i] <- as.numeric(end - start) * 1e6  # Convert to microseconds
    }

    med <- median(times)
    cat(sprintf("n=%d: %.2f us (median of %d runs)\n", n, med, n_reps))
  }
}

cat("\n=== Benchmark Complete ===\n")

# Also test with varying numbers of variables
cat("\n=== Variable Count Scaling (n=1000) ===\n")

n <- 1000
var_configs <- list(
  c(2, 2),
  c(5, 3),
  c(10, 5),
  c(20, 10)
)

for (config in var_configs) {
  p <- config[1]
  q <- config[2]

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  Y <- matrix(rnorm(n * q), nrow = n, ncol = q)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      cancor(X, Y),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
  } else {
    times <- numeric(50)
    for (i in 1:50) {
      start <- Sys.time()
      result <- cancor(X, Y)
      end <- Sys.time()
      times[i] <- as.numeric(end - start) * 1e6
    }
    med <- median(times)
  }

  cat(sprintf("p=%d, q=%d: %.2f us\n", p, q, med))
}
