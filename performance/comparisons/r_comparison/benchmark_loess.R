#!/usr/bin/env Rscript
# LOESS R Benchmark
# Compares R loess() implementation performance

# Check for microbenchmark
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (use_microbenchmark) {
    library(microbenchmark)
}

set.seed(42)

cat("=== LOESS R Benchmarks ===\n\n")

# Generate test data
generate_loess_data <- function(n) {
    x <- seq(0, 4 * pi, length.out = n)
    y <- sin(x) + 0.3 * x + rnorm(n, sd = 0.5)
    list(x = x, y = y)
}

# Benchmark sizes
sizes <- c(100, 1000, 10000)

cat("Gaussian family (default):\n")
for (n in sizes) {
    data <- generate_loess_data(n)

    if (use_microbenchmark) {
        bm <- microbenchmark(
            loess(data$y ~ data$x, span = 0.5, degree = 2, family = "gaussian"),
            times = 20,
            unit = "milliseconds"
        )
        med <- median(bm$time) / 1e6  # Convert ns to ms
        cat(sprintf("  n=%d: %.3f ms (median of 20)\n", n, med))
    } else {
        timing <- system.time(replicate(20, {
            loess(data$y ~ data$x, span = 0.5, degree = 2, family = "gaussian")
        }))
        avg_ms <- timing["elapsed"] * 1000 / 20
        cat(sprintf("  n=%d: %.3f ms (avg of 20)\n", n, avg_ms))
    }
}

cat("\nRobust (symmetric) family:\n")
for (n in c(100, 1000, 5000)) {
    data <- generate_loess_data(n)

    if (use_microbenchmark) {
        bm <- microbenchmark(
            loess(data$y ~ data$x, span = 0.5, degree = 2, family = "symmetric"),
            times = 10,
            unit = "milliseconds"
        )
        med <- median(bm$time) / 1e6
        cat(sprintf("  n=%d: %.3f ms (median of 10)\n", n, med))
    } else {
        timing <- system.time(replicate(10, {
            loess(data$y ~ data$x, span = 0.5, degree = 2, family = "symmetric")
        }))
        avg_ms <- timing["elapsed"] * 1000 / 10
        cat(sprintf("  n=%d: %.3f ms (avg of 10)\n", n, avg_ms))
    }
}

cat("\nSpan comparison (n=1000):\n")
data <- generate_loess_data(1000)
for (span in c(0.3, 0.5, 0.75, 0.9)) {
    if (use_microbenchmark) {
        bm <- microbenchmark(
            loess(data$y ~ data$x, span = span, degree = 2, family = "gaussian"),
            times = 20,
            unit = "milliseconds"
        )
        med <- median(bm$time) / 1e6
        cat(sprintf("  span=%.2f: %.3f ms (median of 20)\n", span, med))
    } else {
        timing <- system.time(replicate(20, {
            loess(data$y ~ data$x, span = span, degree = 2, family = "gaussian")
        }))
        avg_ms <- timing["elapsed"] * 1000 / 20
        cat(sprintf("  span=%.2f: %.3f ms (avg of 20)\n", span, avg_ms))
    }
}

cat("\n=== Benchmark complete ===\n")
