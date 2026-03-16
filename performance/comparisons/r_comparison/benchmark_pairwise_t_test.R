#!/usr/bin/env Rscript
# Pairwise t-test R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Pairwise t-test R Benchmarks ===\n\n")

# Generate test data for different sizes
generate_data <- function(k_groups, n_per_group) {
  # k groups, each with n_per_group observations
  # Group means separated by 10 units to ensure significant differences
  values <- unlist(lapply(1:k_groups, function(g) {
    rnorm(n_per_group, mean = g * 10, sd = 2)
  }))
  groups <- factor(rep(paste0("G", 1:k_groups), each = n_per_group))
  list(values = values, groups = groups)
}

# Test configurations: (k_groups, n_per_group)
configs <- list(
  c(5, 20),     # Small: 5 groups, 20 per group (n=100)
  c(10, 100),   # Medium: 10 groups, 100 per group (n=1000)
  c(10, 1000)   # Large: 10 groups, 1000 per group (n=10000)
)

cat("Benchmarking pairwise.t.test with pool.sd=TRUE, p.adjust.method='holm'\n")
cat("-------------------------------------------------------------------\n\n")

for (cfg in configs) {
  k <- cfg[1]
  n <- cfg[2]
  total_n <- k * n
  n_comparisons <- k * (k - 1) / 2

  data <- generate_data(k, n)

  cat(sprintf("k=%d groups, n=%d per group (total N=%d, %d comparisons):\n",
              k, n, total_n, n_comparisons))

  if (use_microbenchmark) {
    # Run benchmark with microbenchmark
    bm <- microbenchmark(
      pairwise.t.test(data$values, data$groups, pool.sd = TRUE, p.adjust.method = "holm"),
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
      pairwise.t.test(data$values, data$groups, pool.sd = TRUE, p.adjust.method = "holm")
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Benchmarking p.adjust methods ===\n\n")

# Benchmark p.adjust with different numbers of p-values
p_sizes <- c(100, 1000, 10000)

for (n_p in p_sizes) {
  p_values <- runif(n_p, 0, 0.1)  # Random p-values between 0 and 0.1

  cat(sprintf("n=%d p-values:\n", n_p))

  for (method in c("holm", "bonferroni", "BH", "hochberg")) {
    if (use_microbenchmark) {
      bm <- microbenchmark(
        p.adjust(p_values, method = method),
        times = 100,
        unit = "microseconds"
      )
      med <- median(bm$time) / 1000
      cat(sprintf("  %s: %.2f µs\n", method, med))
    } else {
      n_iter <- 1000
      timing <- system.time(for(i in 1:n_iter) p.adjust(p_values, method = method))
      avg_us <- (timing["elapsed"] / n_iter) * 1e6
      cat(sprintf("  %s: %.2f µs\n", method, avg_us))
    }
  }
  cat("\n")
}

cat("=== Validation Output ===\n\n")

# Generate validation data for Rust tests
x <- c(1.0, 2.0, 3.0, 2.5, 1.5, 10.0, 11.0, 12.0, 10.5, 11.5, 20.0, 21.0, 22.0, 20.5, 21.5)
g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))

cat("Test data: 3 groups (A, B, C), 5 observations each\n")
cat("Group A: c(1.0, 2.0, 3.0, 2.5, 1.5), mean =", mean(x[1:5]), "\n")
cat("Group B: c(10.0, 11.0, 12.0, 10.5, 11.5), mean =", mean(x[6:10]), "\n")
cat("Group C: c(20.0, 21.0, 22.0, 20.5, 21.5), mean =", mean(x[11:15]), "\n\n")

cat("pairwise.t.test with pool.sd=TRUE, p.adjust='none':\n")
result <- pairwise.t.test(x, g, pool.sd = TRUE, p.adjust.method = "none")
print(result$p.value)
cat("\n")

cat("pairwise.t.test with pool.sd=TRUE, p.adjust='holm':\n")
result_holm <- pairwise.t.test(x, g, pool.sd = TRUE, p.adjust.method = "holm")
print(result_holm$p.value)
cat("\n")

cat("p.adjust validation:\n")
p <- c(0.001, 0.01, 0.05, 0.1)
cat("Input p-values:", p, "\n")
cat("Holm:       ", p.adjust(p, method = "holm"), "\n")
cat("BH:         ", p.adjust(p, method = "BH"), "\n")
cat("Bonferroni: ", p.adjust(p, method = "bonferroni"), "\n")
cat("Hochberg:   ", p.adjust(p, method = "hochberg"), "\n")
