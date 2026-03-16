#!/usr/bin/env Rscript
# Phillips-Perron Unit Root Test R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  cat("Warning: microbenchmark not available, using system.time fallback\n")
  use_microbenchmark <- FALSE
} else {
  library(microbenchmark)
  use_microbenchmark <- TRUE
}

set.seed(42)

# Generate test data for different sizes
generate_test_data <- function(n) {
  # Generate random walk (typical PP test input)
  cumsum(rnorm(n))
}

# Benchmark at different dataset sizes
sizes <- c(100, 1000, 10000)

cat("=== Phillips-Perron Test R Benchmarks ===\n")
cat("Testing with lshort=TRUE (default)\n\n")

for (n in sizes) {
  x <- generate_test_data(n)

  if (use_microbenchmark) {
    bm <- microbenchmark(
      PP.test(x),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    # Fallback: use system.time with 50 iterations
    timing <- system.time(replicate(50, { PP.test(x) }))
    avg <- timing["elapsed"] * 1000000 / 50  # Convert to microseconds
    cat(sprintf("  n=%d: %.2f us (average of 50 runs)\n", n, avg))
  }
}

cat("\n=== Test with lshort=FALSE ===\n")

for (n in sizes) {
  x <- generate_test_data(n)

  if (use_microbenchmark) {
    bm <- microbenchmark(
      PP.test(x, lshort = FALSE),
      times = 50,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med))
  } else {
    timing <- system.time(replicate(50, { PP.test(x, lshort = FALSE) }))
    avg <- timing["elapsed"] * 1000000 / 50
    cat(sprintf("  n=%d: %.2f us (average of 50 runs)\n", n, avg))
  }
}

cat("\n=== Validation Results ===\n")

# Verify truncation lag calculation
cat("Truncation lag verification:\n")
for (n in c(100, 1000, 10000)) {
  x <- generate_test_data(n)
  result_short <- PP.test(x, lshort = TRUE)
  result_long <- PP.test(x, lshort = FALSE)
  cat(sprintf("  n=%d: lshort=TRUE lag=%d, lshort=FALSE lag=%d\n",
              n, result_short$parameter, result_long$parameter))
}

cat("\nSample output for n=1000:\n")
x <- generate_test_data(1000)
print(PP.test(x))
