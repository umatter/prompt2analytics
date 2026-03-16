#!/usr/bin/env Rscript
# Quade Test R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Quade Test R Benchmarks ===\n\n")

# Generate blocked data: n_blocks x n_treatments matrix
generate_blocked_data <- function(n_blocks, n_treatments) {
  # Each treatment adds to the base effect
  # Block effects add random shift
  data <- matrix(0, nrow = n_blocks, ncol = n_treatments)

  for (i in 1:n_blocks) {
    block_effect <- rnorm(1, mean = 0, sd = 5)  # Random block shift
    for (j in 1:n_treatments) {
      treatment_effect <- (j - 1) * 2  # Treatment effect
      data[i, j] <- block_effect + treatment_effect + rnorm(1, sd = 2)
    }
  }

  colnames(data) <- paste0("T", 1:n_treatments)
  rownames(data) <- paste0("B", 1:n_blocks)

  data
}

# Test configurations: (n_blocks, n_treatments)
configs <- list(
  c(20, 5),    # Small: 20 blocks, 5 treatments (n=100)
  c(200, 5),   # Medium: 200 blocks, 5 treatments (n=1000)
  c(2000, 5)   # Large: 2000 blocks, 5 treatments (n=10000)
)

cat("Benchmarking quade.test with varying block and treatment sizes\n")
cat("-------------------------------------------------------------------\n\n")

for (cfg in configs) {
  n_blocks <- cfg[1]
  n_treatments <- cfg[2]

  data <- generate_blocked_data(n_blocks, n_treatments)

  cat(sprintf("n_blocks=%d, n_treatments=%d:\n", n_blocks, n_treatments))

  if (use_microbenchmark) {
    # Run benchmark with microbenchmark
    bm <- microbenchmark(
      quade.test(data),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  Median: %.2f µs\n", med))
    cat(sprintf("  Mean:   %.2f µs\n", mean(bm$time) / 1000))
    cat(sprintf("  Min:    %.2f µs\n", min(bm$time) / 1000))
    cat(sprintf("  Max:    %.2f µs\n\n", max(bm$time) / 1000))
  } else {
    # Fallback: use system.time with 100 iterations
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      quade.test(data)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Generate validation data for Rust tests (same as in R documentation)
y <- matrix(c(5, 4, 7, 10, 12,
              1, 3, 1, 0, 2,
              16, 12, 22, 22, 35,
              5, 4, 3, 5, 4,
              10, 9, 7, 13, 10,
              19, 18, 28, 25, 20,
              10, 7, 6, 8, 7),
            nrow = 7, byrow = TRUE,
            dimnames = list(Store = as.character(1:7),
                            Brand = LETTERS[1:5]))

cat("Test data: 7 blocks (stores), 5 treatments (brands)\n")
cat("Data matrix:\n")
print(y)
cat("\n")

result <- quade.test(y)
cat("quade.test result:\n")
print(result)

cat("\n\nIntermediate values:\n")
b <- 7
k <- 5

# Block ranges
ranges <- apply(y, 1, function(u) max(u) - min(u))
cat("Block ranges:", ranges, "\n")

# Rank the ranges
q <- rank(ranges)
cat("Ranked ranges (Q):", q, "\n")

# Within-block ranks
r <- t(apply(y, 1, rank))
cat("Within-block ranks:\n")
print(r)

# S matrix
s <- q * (r - (k+1)/2)
cat("\nS matrix:\n")
print(s)

# A and B
A <- sum(s^2)
B <- sum(colSums(s)^2) / b
cat("\nA =", A, ", B =", B, "\n")
cat("F = (b-1)*B / (A-B) =", (b-1)*B / (A-B), "\n")
