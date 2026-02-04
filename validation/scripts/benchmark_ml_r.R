#!/usr/bin/env Rscript
# Benchmark R ML methods for comparison with Rust
# Run with: Rscript validation/scripts/benchmark_ml_r.R

library(e1071)
library(rpart)

cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("R ML BENCHMARKS\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n\n")

# Test different data sizes
sizes <- c(1000, 5000, 10000, 20000)

for (n in sizes) {
  cat(sprintf("\n--- n = %d ---\n", n))

  set.seed(42)
  x1 <- runif(n)
  x2 <- runif(n)
  x3 <- rnorm(n, 0, 0.5)
  y_reg <- 2 * x1 + 0.5 * x2 + rnorm(n, sd=0.3)
  y_class <- factor(ifelse(x1 + x2 > 1, 1, 0))

  df <- data.frame(y_reg=y_reg, y_class=y_class, x1=x1, x2=x2, x3=x3)
  X <- as.matrix(df[, c("x1", "x2", "x3")])

  # SVM RBF
  t0 <- Sys.time()
  svm_fit <- svm(y_class ~ x1 + x2 + x3, data=df, kernel="radial",
                 gamma=1/3, cost=1.0, scale=FALSE)
  t_svm <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("SVM RBF:    %.4f sec\n", t_svm))

  # SVM Linear
  t0 <- Sys.time()
  svm_lin <- svm(y_class ~ x1 + x2 + x3, data=df, kernel="linear",
                 cost=1.0, scale=FALSE)
  t_svm_lin <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("SVM Linear: %.4f sec\n", t_svm_lin))

  # CART Regression
  t0 <- Sys.time()
  cart_reg <- rpart(y_reg ~ x1 + x2 + x3, data=df, method="anova",
                    control=rpart.control(maxdepth=5, minsplit=10, cp=0.01))
  t_cart_reg <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("CART Reg:   %.4f sec\n", t_cart_reg))

  # CART Classification
  t0 <- Sys.time()
  cart_class <- rpart(y_class ~ x1 + x2 + x3, data=df, method="class",
                      control=rpart.control(maxdepth=5, minsplit=10, cp=0.01))
  t_cart_class <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("CART Class: %.4f sec\n", t_cart_class))

  # ROC/AUC calculation
  pred_probs <- predict(cart_class, df)[, 2]
  y_true <- as.numeric(df$y_class) - 1

  t0 <- Sys.time()
  # Manual AUC calculation (same as validation script)
  ord <- order(pred_probs, decreasing=TRUE)
  labels <- y_true[ord]
  n_pos <- sum(labels == 1)
  n_neg <- sum(labels == 0)
  tpr <- cumsum(labels) / n_pos
  fpr <- cumsum(1 - labels) / n_neg
  auc <- sum(diff(c(0, fpr)) * (c(0, tpr)[-1] + c(0, tpr)[-length(c(0, tpr))]) / 2)
  t_auc <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  cat(sprintf("ROC/AUC:    %.4f sec\n", t_auc))
}

cat("\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("BENCHMARK COMPLETE\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n")
