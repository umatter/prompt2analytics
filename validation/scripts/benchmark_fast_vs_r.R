#!/usr/bin/env Rscript
# Benchmark R implementations for comparison with Rust fast implementations
# Run with: Rscript validation/scripts/benchmark_fast_vs_r.R

library(e1071)
# library(pROC)  # Optional

cat("============================================================\n")
cat("R BENCHMARK: SVM and ROC/AUC\n")
cat("============================================================\n\n")

# Deterministic data generation matching Rust
generate_data <- function(n) {
  x <- matrix(0, n, 3)
  y_class <- numeric(n)
  predictions <- numeric(n)
  actual <- numeric(n)

  for (i in 1:n) {
    idx <- i - 1  # 0-indexed like Rust
    x1 <- ((idx * 48271) %% 10000) / 10000
    x2 <- ((idx * 16807 + 5000) %% 10000) / 10000
    x3 <- (((idx * 1103515245 + 12345) %% 10000) / 10000 - 0.5) * 0.5

    x[i, 1] <- x1
    x[i, 2] <- x2
    x[i, 3] <- x3

    y_class[i] <- if (x1 + x2 > 1.0) 1 else -1
    predictions[i] <- x1 + x2
    actual[i] <- if (x1 + x2 > 1.0) 1 else 0
  }

  list(x = x, y_class = factor(y_class), predictions = predictions, actual = actual)
}

# ROC/AUC Benchmark using Mann-Whitney U (same algorithm as Rust fast implementation)
cat("ROC/AUC BENCHMARK (Mann-Whitney U in R)\n")
cat("---------------------------------------\n")
cat(sprintf("%10s | %15s\n", "n", "R Mann-Whitney (ms)"))
cat(strrep("-", 35), "\n")

sizes_roc <- c(1000, 5000, 10000, 50000, 100000)
for (n in sizes_roc) {
  data <- generate_data(n)

  # Manual O(n log n) using same algorithm as Rust
  t0 <- Sys.time()
  n_total <- length(data$predictions)
  ord <- order(data$predictions)
  sorted_labels <- data$actual[ord]
  n_pos <- sum(sorted_labels == 1)
  n_neg <- n_total - n_pos

  # Compute ranks (handle ties)
  ranks <- numeric(n_total)
  i <- 1
  while (i <= n_total) {
    j <- i
    pred_val <- data$predictions[ord[i]]
    while (j <= n_total && abs(data$predictions[ord[j]] - pred_val) < 1e-15) {
      j <- j + 1
    }
    avg_rank <- (i + j) / 2
    for (k in i:(j-1)) {
      ranks[k] <- avg_rank
    }
    i <- j
  }

  rank_sum_pos <- sum(ranks[sorted_labels == 1])
  U <- rank_sum_pos - n_pos * (n_pos + 1) / 2
  auc_manual <- U / (n_pos * n_neg)
  t_manual <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000

  cat(sprintf("%10d | %15.2f\n", n, t_manual))
}

cat("\n")

# SVM Benchmark
cat("SVM RBF BENCHMARK\n")
cat("-----------------\n")
cat(sprintf("%10s | %15s\n", "n", "e1071 (ms)"))
cat(strrep("-", 30), "\n")

sizes_svm <- c(100, 500, 1000, 2000)
for (n in sizes_svm) {
  data <- generate_data(n)
  df <- data.frame(y = data$y_class, x1 = data$x[,1], x2 = data$x[,2], x3 = data$x[,3])

  t0 <- Sys.time()
  svm_fit <- svm(y ~ ., data = df, kernel = "radial", gamma = 1/3, cost = 1.0, scale = FALSE)
  t_svm <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000

  cat(sprintf("%10d | %15.2f\n", n, t_svm))
}

cat("\n")

# Linear SVM Benchmark
cat("SVM LINEAR BENCHMARK\n")
cat("--------------------\n")
cat(sprintf("%10s | %15s\n", "n", "e1071 (ms)"))
cat(strrep("-", 30), "\n")

for (n in sizes_svm) {
  data <- generate_data(n)
  df <- data.frame(y = data$y_class, x1 = data$x[,1], x2 = data$x[,2], x3 = data$x[,3])

  t0 <- Sys.time()
  svm_fit <- svm(y ~ ., data = df, kernel = "linear", cost = 1.0, scale = FALSE)
  t_svm <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000

  cat(sprintf("%10d | %15.2f\n", n, t_svm))
}

cat("\n")
cat("============================================================\n")
cat("BENCHMARK COMPLETE\n")
cat("============================================================\n")
