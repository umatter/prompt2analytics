#!/usr/bin/env Rscript
# Kruskal-Wallis Rank Sum Test - R Benchmark
# Compares R implementation performance against p2a Rust
#
# References:
# - Kruskal & Wallis (1952), JASA 47(260):583-621
# - Hollander & Wolfe (1973), Nonparametric Statistical Methods, pp. 115-120

set.seed(42)

# Benchmark at different dataset sizes (total observations, split into 3 groups)
sizes <- c(100, 1000, 10000)

cat("=== Kruskal-Wallis Rank Sum Test R Benchmarks ===\n\n")

# Try to use microbenchmark if available, otherwise fallback to system.time
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (use_microbenchmark) {
  library(microbenchmark)

  for (n in sizes) {
    n_per_group <- n %/% 3

    # Generate 3 groups with different means
    g1 <- runif(n_per_group, 0, 10)
    g2 <- runif(n_per_group, 2, 12)
    g3 <- runif(n_per_group, 4, 14)

    values <- c(g1, g2, g3)
    groups <- factor(c(rep("G1", n_per_group), rep("G2", n_per_group), rep("G3", n_per_group)))

    # Warmup
    invisible(kruskal.test(values ~ groups))

    # Benchmark
    bm <- microbenchmark(
      kruskal.test(values ~ groups),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  n=%d: %.2f us (median of 100 runs)\n", n, med))
  }
} else {
  cat("Note: microbenchmark not available, using system.time fallback\n\n")

  for (n in sizes) {
    n_per_group <- n %/% 3

    g1 <- runif(n_per_group, 0, 10)
    g2 <- runif(n_per_group, 2, 12)
    g3 <- runif(n_per_group, 4, 14)

    values <- c(g1, g2, g3)
    groups <- factor(c(rep("G1", n_per_group), rep("G2", n_per_group), rep("G3", n_per_group)))

    # Warmup
    invisible(kruskal.test(values ~ groups))

    # Benchmark with 50 replications
    timing <- system.time(replicate(50, { kruskal.test(values ~ groups) }))
    med_ms <- timing["elapsed"] * 1000 / 50
    med_us <- med_ms * 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test 1: Basic three-group comparison
cat("Test 1: Three groups with clear differences\n")
x1 <- c(2.9, 3.0, 2.5, 2.6, 3.2)
x2 <- c(3.8, 2.7, 4.0, 2.4)
x3 <- c(2.8, 3.4, 3.7, 2.2, 2.0)
result1 <- kruskal.test(list(x1, x2, x3))
cat(sprintf("  H = %.4f, df = %d, p-value = %.4f\n", result1$statistic, result1$parameter, result1$p.value))

# Test 2: airquality-like data
cat("\nTest 2: Simulated airquality-like data\n")
may <- c(41, 36, 12, 18, 23)
jun <- c(29, 45, 71, 39, 32)
jul <- c(135, 49, 32, 64, 40)
result2 <- kruskal.test(list(May = may, Jun = jun, Jul = jul))
cat(sprintf("  H = %.4f, df = %d, p-value = %.4f\n", result2$statistic, result2$parameter, result2$p.value))

# Test 3: Manual calculation verification
cat("\nTest 3: Simple 3-group test (manual verification)\n")
g1 <- c(1, 2)
g2 <- c(3, 4)
g3 <- c(5, 6)
result3 <- kruskal.test(list(A = g1, B = g2, C = g3))
cat(sprintf("  Groups: A = c(1,2), B = c(3,4), C = c(5,6)\n"))
cat(sprintf("  H = %.4f, df = %d, p-value = %.4f\n", result3$statistic, result3$parameter, result3$p.value))

# Test 4: Data with ties
cat("\nTest 4: Data with ties\n")
t1 <- c(1, 2, 2, 3)
t2 <- c(2, 3, 3, 4)
t3 <- c(3, 4, 4, 5)
result4 <- kruskal.test(list(t1, t2, t3))
cat(sprintf("  H = %.4f, df = %d, p-value = %.4f\n", result4$statistic, result4$parameter, result4$p.value))

cat("\nDone.\n")
