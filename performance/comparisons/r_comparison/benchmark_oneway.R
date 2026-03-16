#!/usr/bin/env Rscript
# Welch's One-Way ANOVA Test - R Benchmark
# Compares R implementation performance against p2a Rust
#
# References:
# - Welch (1951), Biometrika 38(3/4):330-336

set.seed(42)

# Benchmark at different dataset sizes (total observations, split into 3 groups)
sizes <- c(100, 1000, 10000)

cat("=== Welch's One-Way ANOVA R Benchmarks ===\n\n")

# Try to use microbenchmark if available
use_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (use_microbenchmark) {
  library(microbenchmark)

  for (n in sizes) {
    n_per_group <- n %/% 3
    g1 <- rnorm(n_per_group, mean = 0, sd = 1)
    g2 <- rnorm(n_per_group, mean = 2, sd = 2)
    g3 <- rnorm(n_per_group, mean = 1, sd = 1.5)

    values <- c(g1, g2, g3)
    groups <- factor(c(rep("A", n_per_group), rep("B", n_per_group), rep("C", n_per_group)))
    dat <- data.frame(value = values, group = groups)

    # Warmup
    invisible(oneway.test(value ~ group, data = dat, var.equal = FALSE))

    # Benchmark
    bm <- microbenchmark(
      oneway.test(value ~ group, data = dat, var.equal = FALSE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("  n=%d: %.2f us (median of 100 runs)\n", n, med))
  }
} else {
  cat("Note: microbenchmark not available, using system.time fallback\n\n")

  for (n in sizes) {
    n_per_group <- n %/% 3
    g1 <- rnorm(n_per_group, mean = 0, sd = 1)
    g2 <- rnorm(n_per_group, mean = 2, sd = 2)
    g3 <- rnorm(n_per_group, mean = 1, sd = 1.5)

    values <- c(g1, g2, g3)
    groups <- factor(c(rep("A", n_per_group), rep("B", n_per_group), rep("C", n_per_group)))
    dat <- data.frame(value = values, group = groups)

    # Warmup
    invisible(oneway.test(value ~ group, data = dat, var.equal = FALSE))

    # Benchmark with 50 replications
    timing <- system.time(replicate(50, { oneway.test(value ~ group, data = dat, var.equal = FALSE) }))
    med_ms <- timing["elapsed"] * 1000 / 50
    med_us <- med_ms * 1000
    cat(sprintf("  n=%d: %.2f us (median of 50 runs)\n", n, med_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test 1: Basic Welch ANOVA
cat("Test 1: Welch ANOVA (var.equal = FALSE)\n")
x <- c(1, 2, 3, 4, 5)
y <- c(10, 11, 12, 13, 14, 15)
z <- c(3, 4, 5)
g <- factor(c(rep("A", 5), rep("B", 6), rep("C", 3)))
dat <- data.frame(value = c(x, y, z), group = g)
result1 <- oneway.test(value ~ group, data = dat, var.equal = FALSE)
cat(sprintf("  F = %.4f, df1 = %.0f, df2 = %.4f, p-value = %.6f\n",
            result1$statistic, result1$parameter[1], result1$parameter[2], result1$p.value))

# Test 2: Standard ANOVA
cat("\nTest 2: Standard ANOVA (var.equal = TRUE)\n")
result2 <- oneway.test(value ~ group, data = dat, var.equal = TRUE)
cat(sprintf("  F = %.4f, df1 = %.0f, df2 = %.0f, p-value = %.6f\n",
            result2$statistic, result2$parameter[1], result2$parameter[2], result2$p.value))

cat("\nDone.\n")
