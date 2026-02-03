#!/usr/bin/env Rscript
# cor.test R Benchmark
# Compares R implementation performance against p2a Rust

library(microbenchmark)

set.seed(42)

cat("=== cor.test R Benchmarks ===\n")

sizes <- c(100, 1000, 10000)

for (n in sizes) {
  x <- rnorm(n)
  y <- 0.7 * x + rnorm(n, sd = 0.5)

  bm_pearson <- microbenchmark(
    cor.test(x, y, method = "pearson"),
    times = 100,
    unit = "microseconds"
  )

  bm_spearman <- microbenchmark(
    cor.test(x, y, method = "spearman"),
    times = 100,
    unit = "microseconds"
  )

  bm_kendall <- microbenchmark(
    cor.test(x, y, method = "kendall"),
    times = 50,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d:\n", n))
  cat(sprintf("    Pearson:  %.2f us (median)\n", median(bm_pearson$time) / 1000))
  cat(sprintf("    Spearman: %.2f us (median)\n", median(bm_spearman$time) / 1000))
  cat(sprintf("    Kendall:  %.2f us (median)\n", median(bm_kendall$time) / 1000))
}

# Validation test
cat("\n=== Validation ===\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(1.2, 2.1, 2.9, 4.1, 5.0, 5.9, 7.1, 7.9, 9.0, 10.1)

result <- cor.test(x, y)
cat(sprintf("Pearson r: %.6f\n", result$estimate))
cat(sprintf("t-stat: %.6f\n", result$statistic))
cat(sprintf("p-value: %.10f\n", result$p.value))
cat(sprintf("95%% CI: [%.6f, %.6f]\n", result$conf.int[1], result$conf.int[2]))
