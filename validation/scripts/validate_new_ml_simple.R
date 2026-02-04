#!/usr/bin/env Rscript
# Simplified validation for newly implemented ML methods
# Uses only available packages: e1071, rpart

cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("ML METHODS VALIDATION: R Reference Values\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

# ============================================================
# Create test data
# ============================================================
set.seed(42)
n <- 200
x1 <- runif(n, 0, 1)
x2 <- runif(n, 0, 1)
x3 <- rnorm(n, 0, 0.5)  # noise variable

# Regression: y = 2*x1 + 0.5*x2 + noise (x1 is most important)
y_reg <- 2 * x1 + 0.5 * x2 + rnorm(n, sd=0.3)

# Classification: based on x1 + x2 > 1
prob <- plogis(3 * (x1 + x2 - 1))
y_class <- rbinom(n, 1, prob)

df <- data.frame(y_reg=y_reg, y_class=factor(y_class), x1=x1, x2=x2, x3=x3)
X <- as.matrix(df[, c("x1", "x2", "x3")])

cat("Test Data:\n")
cat(sprintf("  n = %d, p = %d\n", n, ncol(X)))
cat(sprintf("  y_reg: mean=%.4f, sd=%.4f\n", mean(y_reg), sd(y_reg)))
cat(sprintf("  y_class: %.1f%% positive\n\n", mean(as.numeric(y_class)-1)*100))

# ============================================================
# 1. KERNEL SVM (e1071)
# ============================================================
library(e1071)

cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("1. KERNEL SVM (e1071::svm)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# RBF Kernel
svm_rbf <- svm(y_class ~ x1 + x2 + x3, data=df,
               type="C-classification", kernel="radial",
               gamma=1/3, cost=1.0, scale=FALSE)
pred_rbf <- predict(svm_rbf, df)
acc_rbf <- mean(pred_rbf == df$y_class)

cat("RBF Kernel (C=1, gamma=0.333):\n")
cat(sprintf("  Accuracy: %.4f\n", acc_rbf))
cat(sprintf("  Support Vectors: %d (%.1f%%)\n\n",
            nrow(svm_rbf$SV), 100*nrow(svm_rbf$SV)/n))

# Linear Kernel
svm_linear <- svm(y_class ~ x1 + x2 + x3, data=df,
                  type="C-classification", kernel="linear",
                  cost=1.0, scale=FALSE)
pred_linear <- predict(svm_linear, df)
acc_linear <- mean(pred_linear == df$y_class)

cat("Linear Kernel (C=1):\n")
cat(sprintf("  Accuracy: %.4f\n", acc_linear))
cat(sprintf("  Support Vectors: %d\n\n", nrow(svm_linear$SV)))

# Polynomial Kernel
svm_poly <- svm(y_class ~ x1 + x2 + x3, data=df,
                type="C-classification", kernel="polynomial",
                degree=3, gamma=1/3, coef0=0, cost=1.0, scale=FALSE)
pred_poly <- predict(svm_poly, df)
acc_poly <- mean(pred_poly == df$y_class)

cat("Polynomial Kernel (degree=3, C=1):\n")
cat(sprintf("  Accuracy: %.4f\n", acc_poly))
cat(sprintf("  Support Vectors: %d\n\n", nrow(svm_poly$SV)))

# ============================================================
# 2. CART (rpart)
# ============================================================
library(rpart)

cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("2. CART Decision Trees (rpart)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# Regression tree
cart_reg <- rpart(y_reg ~ x1 + x2 + x3, data=df, method="anova",
                  control=rpart.control(maxdepth=5, minsplit=10, cp=0.01))
pred_cart <- predict(cart_reg, df)
mse_cart <- mean((y_reg - pred_cart)^2)
r2_cart <- 1 - mse_cart/var(y_reg)

cat("CART Regression (depth=5, cp=0.01):\n")
cat(sprintf("  MSE: %.6f\n", mse_cart))
cat(sprintf("  R²:  %.4f\n", r2_cart))
cat(sprintf("  Nodes: %d\n\n", nrow(cart_reg$frame)))

# Variable importance
if (!is.null(cart_reg$variable.importance)) {
  imp <- cart_reg$variable.importance
  imp_norm <- imp / sum(imp)
  cat("Variable Importance (normalized):\n")
  for (nm in names(imp_norm)) {
    cat(sprintf("  %s: %.4f\n", nm, imp_norm[nm]))
  }
  cat("\n")
}

# Classification tree
cart_class <- rpart(y_class ~ x1 + x2 + x3, data=df, method="class",
                    control=rpart.control(maxdepth=5, minsplit=10, cp=0.01))
pred_cart_class <- predict(cart_class, df, type="class")
acc_cart <- mean(pred_cart_class == df$y_class)

cat("CART Classification (Gini):\n")
cat(sprintf("  Accuracy: %.4f\n\n", acc_cart))

# ============================================================
# 3. ROC/AUC (manual calculation)
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("3. ROC/AUC (manual calculation)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# Get class probabilities from CART
pred_probs <- predict(cart_class, df)[, 2]  # probability of class 1
y_true <- as.numeric(df$y_class) - 1

# Calculate AUC using trapezoidal rule
calc_auc <- function(probs, labels) {
  ord <- order(probs, decreasing=TRUE)
  probs <- probs[ord]
  labels <- labels[ord]

  n_pos <- sum(labels == 1)
  n_neg <- sum(labels == 0)

  tpr <- cumsum(labels) / n_pos
  fpr <- cumsum(1 - labels) / n_neg

  # Prepend (0,0)
  tpr <- c(0, tpr)
  fpr <- c(0, fpr)

  # Trapezoidal rule
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  return(auc)
}

auc_val <- calc_auc(pred_probs, y_true)
cat(sprintf("AUC (from CART probs): %.4f\n", auc_val))

# Optimal threshold (Youden's J)
thresholds <- sort(unique(pred_probs))
best_j <- -1
best_thresh <- 0.5
best_sens <- 0
best_spec <- 0

for (t in thresholds) {
  pred_t <- ifelse(pred_probs >= t, 1, 0)
  tp <- sum(pred_t == 1 & y_true == 1)
  tn <- sum(pred_t == 0 & y_true == 0)
  fp <- sum(pred_t == 1 & y_true == 0)
  fn <- sum(pred_t == 0 & y_true == 1)

  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  j <- sens + spec - 1

  if (j > best_j) {
    best_j <- j
    best_thresh <- t
    best_sens <- sens
    best_spec <- spec
  }
}

cat(sprintf("Optimal Threshold (Youden): %.4f\n", best_thresh))
cat(sprintf("  Sensitivity: %.4f\n", best_sens))
cat(sprintf("  Specificity: %.4f\n\n", best_spec))

# Confusion matrix at threshold=0.5
pred_05 <- ifelse(pred_probs >= 0.5, 1, 0)
tp <- sum(pred_05 == 1 & y_true == 1)
tn <- sum(pred_05 == 0 & y_true == 0)
fp <- sum(pred_05 == 1 & y_true == 0)
fn <- sum(pred_05 == 0 & y_true == 1)

precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

cat("Confusion Matrix (threshold=0.5):\n")
cat(sprintf("  TP=%d, TN=%d, FP=%d, FN=%d\n", tp, tn, fp, fn))
cat(sprintf("  Precision: %.4f\n", precision))
cat(sprintf("  Recall:    %.4f\n", recall))
cat(sprintf("  F1:        %.4f\n\n", f1))

# ============================================================
# SUMMARY
# ============================================================
cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("SUMMARY: REFERENCE VALUES FOR RUST COMPARISON\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

cat("Kernel SVM (all should be >0.80 accuracy):\n")
cat(sprintf("  RBF:        %.4f\n", acc_rbf))
cat(sprintf("  Linear:     %.4f\n", acc_linear))
cat(sprintf("  Polynomial: %.4f\n\n", acc_poly))

cat("CART:\n")
cat(sprintf("  Regression R²: %.4f\n", r2_cart))
cat(sprintf("  Classification Acc: %.4f\n\n", acc_cart))

cat("ROC/AUC:\n")
cat(sprintf("  AUC: %.4f (should be >0.5 for better than random)\n", auc_val))
cat(sprintf("  x1 should be most important (coef=2 vs x2 coef=0.5)\n\n"))

cat("Variable Importance:\n")
cat(sprintf("  Ranking: x1 > x2 > x3 (expected)\n"))
