#!/usr/bin/env Rscript
# Pairwise Wilcoxon R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Pairwise Wilcoxon R Benchmarks ===\n\n")

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
  c(3, 10),     # Small: 3 groups, 10 per group, 3 comparisons
  c(5, 50),     # Medium: 5 groups, 50 per group, 10 comparisons
  c(10, 100),   # Large: 10 groups, 100 per group, 45 comparisons
  c(20, 200)    # Very large: 20 groups, 200 per group, 190 comparisons
)

cat("Benchmarking pairwise.wilcox.test with p.adjust.method='holm'\n")
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
      pairwise.wilcox.test(data$values, data$groups, p.adjust.method = "holm", exact = FALSE),
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
      pairwise.wilcox.test(data$values, data$groups, p.adjust.method = "holm", exact = FALSE)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Generate validation data for Rust tests
x <- c(1.0, 2.0, 3.0, 2.5, 1.5, 10.0, 11.0, 12.0, 10.5, 11.5, 20.0, 21.0, 22.0, 20.5, 21.5)
g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))

cat("Test data: 3 groups (A, B, C), 5 observations each\n")
cat("Group A: c(1.0, 2.0, 3.0, 2.5, 1.5), median =", median(x[1:5]), "\n")
cat("Group B: c(10.0, 11.0, 12.0, 10.5, 11.5), median =", median(x[6:10]), "\n")
cat("Group C: c(20.0, 21.0, 22.0, 20.5, 21.5), median =", median(x[11:15]), "\n\n")

cat("pairwise.wilcox.test with p.adjust='none', exact=FALSE:\n")
result <- pairwise.wilcox.test(x, g, p.adjust.method = "none", exact = FALSE)
print(result$p.value)
cat("\n")

cat("pairwise.wilcox.test with p.adjust='holm', exact=FALSE:\n")
result_holm <- pairwise.wilcox.test(x, g, p.adjust.method = "holm", exact = FALSE)
print(result_holm$p.value)
cat("\n")

# Small sample exact test
cat("Small sample exact test (k=2, n=3):\n")
x_small <- c(1, 2, 3, 4, 5, 6)
g_small <- factor(c("A", "A", "A", "B", "B", "B"))
result_exact <- pairwise.wilcox.test(x_small, g_small, p.adjust.method = "none", exact = TRUE)
print(result_exact$p.value)
