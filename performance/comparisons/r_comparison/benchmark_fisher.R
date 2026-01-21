#!/usr/bin/env Rscript
# Fisher's Exact Test R Benchmark
# Compares R implementation performance against p2a Rust
#
# Run with: Rscript performance/comparisons/r_comparison/benchmark_fisher.R

library(microbenchmark)

set.seed(42)

# ============================================================================
# Data Generation (matching Rust benchmark DGP)
# ============================================================================

generate_2x2_table <- function(total, imbalance = 0.3) {
  base <- total / 4
  a <- max(1, round(base * (1 + imbalance) + runif(1, -5, 5)))
  b <- max(1, round(base * (1 - imbalance * 0.5) + runif(1, -5, 5)))
  c <- max(1, round(base * (1 - imbalance * 0.5) + runif(1, -5, 5)))
  d <- max(1, round(base * (1 + imbalance) + runif(1, -5, 5)))
  matrix(c(a, b, c, d), nrow = 2, byrow = TRUE)
}

# ============================================================================
# Fisher's Exact Test Benchmarks
# ============================================================================

cat("======================================================\n")
cat("R Fisher's Exact Test Benchmarks\n")
cat("======================================================\n")

cat("\n=== Fisher's Exact Test (Two-Sided) ===\n")
results <- list()

for (total in c(20, 100, 500, 1000)) {
  cat(sprintf("  n=%d: ", total))
  table <- generate_2x2_table(total)

  bm <- microbenchmark(
    fisher.test(table, alternative = "two.sided"),
    times = 100,
    unit = "microseconds"
  )

  med <- median(bm$time) / 1000  # Convert to microseconds
  cat(sprintf("%.2f us (median)\n", med))
  results[[paste0("fisher_n", total)]] <- summary(bm)
}

cat("\n=== Fisher's Exact Test with CI ===\n")
for (total in c(20, 100, 500)) {
  cat(sprintf("  n=%d: ", total))
  table <- generate_2x2_table(total)

  bm <- microbenchmark(
    fisher.test(table, alternative = "two.sided", conf.int = TRUE, conf.level = 0.95),
    times = 100,
    unit = "microseconds"
  )

  med <- median(bm$time) / 1000
  cat(sprintf("%.2f us (median)\n", med))
  results[[paste0("fisher_ci_n", total)]] <- summary(bm)
}

cat("\n=== Fisher's Exact Test Alternatives ===\n")
table <- generate_2x2_table(100)
cat("Table:\n")
print(table)

cat("\n  Two-sided: ")
bm_two <- microbenchmark(
  fisher.test(table, alternative = "two.sided"),
  times = 100,
  unit = "microseconds"
)
cat(sprintf("%.2f us (median)\n", median(bm_two$time) / 1000))

cat("  Greater: ")
bm_greater <- microbenchmark(
  fisher.test(table, alternative = "greater"),
  times = 100,
  unit = "microseconds"
)
cat(sprintf("%.2f us (median)\n", median(bm_greater$time) / 1000))

cat("  Less: ")
bm_less <- microbenchmark(
  fisher.test(table, alternative = "less"),
  times = 100,
  unit = "microseconds"
)
cat(sprintf("%.2f us (median)\n", median(bm_less$time) / 1000))

# ============================================================================
# Summary
# ============================================================================

cat("\n======================================================\n")
cat("SUMMARY TABLE (median times in microseconds)\n")
cat("======================================================\n\n")

cat("Fisher's Exact Test (Two-Sided):\n")
for (total in c(20, 100, 500, 1000)) {
  key <- paste0("fisher_n", total)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", total, med))
  }
}

cat("\nFisher's Exact Test with 95% CI:\n")
for (total in c(20, 100, 500)) {
  key <- paste0("fisher_ci_n", total)
  if (!is.null(results[[key]])) {
    med <- results[[key]]$median
    cat(sprintf("  n=%4d: %10.2f us\n", total, med))
  }
}

cat("\nAlternative Hypotheses (n=100):\n")
cat(sprintf("  Two-sided: %10.2f us\n", median(bm_two$time) / 1000))
cat(sprintf("  Greater:   %10.2f us\n", median(bm_greater$time) / 1000))
cat(sprintf("  Less:      %10.2f us\n", median(bm_less$time) / 1000))

cat("\n======================================================\n")
cat("Done. Run Rust benchmarks with:\n")
cat("  cargo bench -p p2a-core -- fisher\n")
cat("======================================================\n")
