#!/usr/bin/env Rscript
# Mahalanobis Distance R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Mahalanobis Distance R Benchmarks ===\n\n")

# Generate correlated multivariate data
generate_data <- function(n, p) {
  data <- matrix(rnorm(n * p), nrow = n, ncol = p)
  # Add correlation between adjacent variables
  for (j in 2:p) {
    data[, j] <- 0.5 * data[, j - 1] + rnorm(n)
  }
  data
}

cat("Sample Size Scaling (p=5)\n")
cat("--------------------------\n\n")

for (n in c(100, 1000, 10000, 100000)) {
  x <- generate_data(n, 5)
  center <- colMeans(x)
  cov_mat <- cov(x)

  cat(sprintf("n=%d:\n", n))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      mahalanobis(x, center, cov_mat),
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
      mahalanobis(x, center, cov_mat)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\nVariable Scaling (n=1000)\n")
cat("--------------------------\n\n")

for (p in c(2, 5, 10, 20, 50)) {
  x <- generate_data(1000, p)
  center <- colMeans(x)
  cov_mat <- cov(x)

  cat(sprintf("p=%d:\n", p))

  if (use_microbenchmark) {
    bm <- microbenchmark(
      mahalanobis(x, center, cov_mat),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  Median: %.2f µs\n\n", med))
  } else {
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      mahalanobis(x, center, cov_mat)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Small example for validation
x <- matrix(c(1, 2, 3, 5, 2, 4, 5, 3), ncol = 2, byrow = TRUE)
center <- colMeans(x)
cov_mat <- cov(x)

cat("Data:\n")
print(x)
cat("\nCenter:\n")
print(center)
cat("\nCovariance matrix:\n")
print(cov_mat)
cat("\nMahalanobis distances squared:\n")
print(mahalanobis(x, center, cov_mat))

# Chi-squared test for multivariate normality
cat("\nChi-squared expected values (df=2):\n")
cat(sprintf("  Mean: %.2f (theoretical: %.2f)\n", mean(mahalanobis(x, center, cov_mat)), 2))
