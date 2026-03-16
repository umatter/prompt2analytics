#!/usr/bin/env Rscript
# Hypothesis Testing Benchmarks for Cross-Language Comparison
# Compares R's t.test() and aov() against p2a Rust implementation

library(microbenchmark)

# Ensure reproducibility
set.seed(42)

# ============================================================================
# Data Generation Functions (matching Rust benchmark DGP)
# ============================================================================

generate_ttest_data <- function(n) {
  # Two groups with different means (matching Rust benchmark)
  x <- 5.0 + runif(n, -1, 1)
  y <- 6.0 + runif(n, -1, 1) * 1.2  # Different mean and variance
  list(x = x, y = y)
}

generate_anova_data <- function(n_per_group, n_groups) {
  y <- numeric()
  group <- character()

  for (g in 1:n_groups) {
    group_mean <- g * 5.0  # Means: 5, 10, 15, ...
    y <- c(y, group_mean + runif(n_per_group, -1, 1))
    group <- c(group, rep(paste0("Group", g-1), n_per_group))
  }

  data.frame(y = y, group = factor(group))
}

generate_two_way_anova_data <- function(n_per_cell, levels_a, levels_b) {
  y <- numeric()
  factor_a <- character()
  factor_b <- character()

  for (a in 1:levels_a) {
    for (b in 1:levels_b) {
      cell_mean <- a * 5.0 + b * 10.0
      y <- c(y, cell_mean + runif(n_per_cell, -1, 1))
      factor_a <- c(factor_a, rep(paste0("A", a-1), n_per_cell))
      factor_b <- c(factor_b, rep(paste0("B", b-1), n_per_cell))
    }
  }

  data.frame(y = y, factor_a = factor(factor_a), factor_b = factor(factor_b))
}

# ============================================================================
# T-Test Benchmarks
# ============================================================================

benchmark_one_sample_ttest <- function() {
  cat("\n=== One-Sample T-Test Benchmarks ===\n")
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("  n=%d: ", n))
    data <- generate_ttest_data(n)

    bm <- microbenchmark(
      t.test(data$x, mu = 5.0),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000  # Convert to microseconds
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("ttest_one_sample_", n)]] <- summary(bm)
  }

  results
}

benchmark_two_sample_ttest <- function() {
  cat("\n=== Two-Sample T-Test (Welch) Benchmarks ===\n")
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("  n=%d per group: ", n))
    data <- generate_ttest_data(n)

    bm <- microbenchmark(
      t.test(data$x, data$y, var.equal = FALSE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("ttest_two_sample_", n)]] <- summary(bm)
  }

  results
}

benchmark_paired_ttest <- function() {
  cat("\n=== Paired T-Test Benchmarks ===\n")
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("  n=%d pairs: ", n))
    data <- generate_ttest_data(n)

    bm <- microbenchmark(
      t.test(data$x, data$y, paired = TRUE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("ttest_paired_", n)]] <- summary(bm)
  }

  results
}

# ============================================================================
# ANOVA Benchmarks
# ============================================================================

benchmark_one_way_anova <- function() {
  cat("\n=== One-Way ANOVA Benchmarks ===\n")
  results <- list()

  configs <- list(
    c(50, 2),     # 100 total
    c(200, 5),    # 1000 total
    c(1000, 10)   # 10000 total
  )

  for (cfg in configs) {
    n_per_group <- cfg[1]
    n_groups <- cfg[2]
    total_n <- n_per_group * n_groups

    cat(sprintf("  n=%d (groups=%d, per_group=%d): ", total_n, n_groups, n_per_group))
    data <- generate_anova_data(n_per_group, n_groups)

    bm <- microbenchmark(
      aov(y ~ group, data = data),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("anova_one_way_n", total_n, "_g", n_groups)]] <- summary(bm)
  }

  results
}

benchmark_two_way_anova <- function() {
  cat("\n=== Two-Way ANOVA Benchmarks ===\n")
  results <- list()

  # 2x2 factorial with varying cell sizes
  for (n_per_cell in c(25, 250, 2500)) {
    total_n <- n_per_cell * 4
    cat(sprintf("  n=%d (2x2, per_cell=%d): ", total_n, n_per_cell))
    data <- generate_two_way_anova_data(n_per_cell, 2, 2)

    bm <- microbenchmark(
      aov(y ~ factor_a * factor_b, data = data),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("anova_two_way_n", total_n, "_2x2")]] <- summary(bm)
  }

  results
}

# ============================================================================
# Chi-Squared Test Benchmarks
# ============================================================================

generate_categorical_data <- function(n_categories, total_count) {
  # Generate random counts for each category (matching Rust benchmark)
  counts <- runif(n_categories, 1, 100)
  counts <- counts / sum(counts) * total_count
  counts
}

generate_contingency_table <- function(n_rows, n_cols, total_count) {
  # Generate random counts for each cell
  table <- matrix(runif(n_rows * n_cols, 1, 100), nrow = n_rows, ncol = n_cols)
  table <- table / sum(table) * total_count
  table
}

benchmark_chisq_gof <- function() {
  cat("\n=== Chi-Squared Goodness-of-Fit Benchmarks ===\n")
  results <- list()

  for (k in c(5, 10, 20, 50, 100)) {
    cat(sprintf("  k=%d categories: ", k))
    observed <- generate_categorical_data(k, 10000)

    bm <- microbenchmark(
      chisq.test(observed),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("chisq_gof_k", k)]] <- summary(bm)
  }

  results
}

benchmark_chisq_independence <- function() {
  cat("\n=== Chi-Squared Independence Test Benchmarks ===\n")
  results <- list()

  for (dims in list(c(2, 2), c(3, 3), c(5, 5), c(10, 10), c(20, 20))) {
    n_rows <- dims[1]
    n_cols <- dims[2]
    cat(sprintf("  %dx%d table: ", n_rows, n_cols))
    table <- generate_contingency_table(n_rows, n_cols, 10000)

    bm <- microbenchmark(
      chisq.test(table, correct = FALSE),
      times = 100,
      unit = "microseconds"
    )

    med <- median(bm$time) / 1000
    cat(sprintf("%.2f us (median)\n", med))
    results[[paste0("chisq_ind_", n_rows, "x", n_cols)]] <- summary(bm)
  }

  results
}

benchmark_chisq_yates <- function() {
  cat("\n=== Chi-Squared 2x2 with Yates Correction Benchmarks ===\n")
  results <- list()

  table <- generate_contingency_table(2, 2, 100)

  cat("  Without Yates: ")
  bm1 <- microbenchmark(
    chisq.test(table, correct = FALSE),
    times = 100,
    unit = "microseconds"
  )
  med1 <- median(bm1$time) / 1000
  cat(sprintf("%.2f us (median)\n", med1))
  results[["chisq_2x2_no_yates"]] <- summary(bm1)

  cat("  With Yates: ")
  bm2 <- microbenchmark(
    chisq.test(table, correct = TRUE),
    times = 100,
    unit = "microseconds"
  )
  med2 <- median(bm2$time) / 1000
  cat(sprintf("%.2f us (median)\n", med2))
  results[["chisq_2x2_with_yates"]] <- summary(bm2)

  results
}

# ============================================================================
# Main Execution
# ============================================================================

cat("======================================================\n")
cat("R Hypothesis Testing Benchmarks\n")
cat("======================================================\n")

# Run all benchmarks
ttest_one_results <- benchmark_one_sample_ttest()
ttest_two_results <- benchmark_two_sample_ttest()
ttest_paired_results <- benchmark_paired_ttest()
anova_one_results <- benchmark_one_way_anova()
anova_two_results <- benchmark_two_way_anova()
chisq_gof_results <- benchmark_chisq_gof()
chisq_ind_results <- benchmark_chisq_independence()
chisq_yates_results <- benchmark_chisq_yates()

# Summary table
cat("\n======================================================\n")
cat("SUMMARY TABLE (median times in microseconds)\n")
cat("======================================================\n")

cat("\nT-Test (One-Sample):\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("ttest_one_sample_", n)
  if (!is.null(ttest_one_results[[key]])) {
    med <- ttest_one_results[[key]]$median
    cat(sprintf("  n=%6d: %10.2f us\n", n, med))
  }
}

cat("\nT-Test (Two-Sample Welch):\n")
for (n in c(100, 1000, 10000)) {
  key <- paste0("ttest_two_sample_", n)
  if (!is.null(ttest_two_results[[key]])) {
    med <- ttest_two_results[[key]]$median
    cat(sprintf("  n=%6d: %10.2f us\n", n, med))
  }
}

cat("\nOne-Way ANOVA:\n")
for (cfg in list(c(100,2), c(1000,5), c(10000,10))) {
  key <- paste0("anova_one_way_n", cfg[1], "_g", cfg[2])
  if (!is.null(anova_one_results[[key]])) {
    med <- anova_one_results[[key]]$median
    cat(sprintf("  n=%5d (g=%2d): %10.2f us\n", cfg[1], cfg[2], med))
  }
}

cat("\nTwo-Way ANOVA:\n")
for (n in c(100, 1000, 10000)) {
  pattern <- paste0("anova_two_way_n", n)
  for (key in names(anova_two_results)) {
    if (grepl(pattern, key)) {
      med <- anova_two_results[[key]]$median
      design <- sub(".*_", "", key)
      cat(sprintf("  n=%4d (%s): %10.2f us\n", n, design, med))
    }
  }
}

cat("\nChi-Squared Goodness-of-Fit:\n")
for (k in c(5, 10, 20, 50, 100)) {
  key <- paste0("chisq_gof_k", k)
  if (!is.null(chisq_gof_results[[key]])) {
    med <- chisq_gof_results[[key]]$median
    cat(sprintf("  k=%3d categories: %10.2f us\n", k, med))
  }
}

cat("\nChi-Squared Independence:\n")
for (dims in list(c(2, 2), c(3, 3), c(5, 5), c(10, 10), c(20, 20))) {
  key <- paste0("chisq_ind_", dims[1], "x", dims[2])
  if (!is.null(chisq_ind_results[[key]])) {
    med <- chisq_ind_results[[key]]$median
    cat(sprintf("  %2dx%2d table:     %10.2f us\n", dims[1], dims[2], med))
  }
}

cat("\nChi-Squared 2x2 with Yates:\n")
if (!is.null(chisq_yates_results[["chisq_2x2_no_yates"]])) {
  med <- chisq_yates_results[["chisq_2x2_no_yates"]]$median
  cat(sprintf("  Without Yates: %10.2f us\n", med))
}
if (!is.null(chisq_yates_results[["chisq_2x2_with_yates"]])) {
  med <- chisq_yates_results[["chisq_2x2_with_yates"]]$median
  cat(sprintf("  With Yates:    %10.2f us\n", med))
}

cat("\n======================================================\n")
cat("Done. Run Rust benchmarks with:\n")
cat("  cargo bench -p p2a-core -- ttest\n")
cat("  cargo bench -p p2a-core -- anova\n")
cat("  cargo bench -p p2a-core -- chisq\n")
cat("======================================================\n")
