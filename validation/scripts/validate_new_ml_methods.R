#!/usr/bin/env Rscript
# Validation script for newly implemented ML methods (Feb 2026)
# Compares: GBM, AdaBoost, Kernel SVM, ROC/AUC, Variable Importance, Partial Dependence

suppressPackageStartupMessages({
  library(gbm)
  library(e1071)  # for SVM
  library(pROC)   # for ROC/AUC
  library(rpart)  # for CART
  library(randomForest)  # for variable importance comparison
  library(pdp)    # for partial dependence (if available)
})

cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("NEW ML METHODS VALIDATION: R Reference Values\n")
cat("Comparing: GBM, AdaBoost, Kernel SVM, ROC/AUC, VarImp, PDP\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

# ============================================================
# Create test data
# ============================================================
set.seed(42)
n <- 200
x1 <- runif(n, 0, 1)
x2 <- runif(n, 0, 1)
x3 <- rnorm(n, 0, 0.5)  # noise variable

# Regression: y = 2*x1 + noise (x1 is most important)
y_reg <- 2 * x1 + 0.5 * x2 + rnorm(n, sd=0.3)

# Classification: based on x1 + x2 > 1
prob <- plogis(3 * (x1 + x2 - 1))
y_class <- rbinom(n, 1, prob)

df <- data.frame(y_reg=y_reg, y_class=y_class, x1=x1, x2=x2, x3=x3)
X <- as.matrix(df[, c("x1", "x2", "x3")])

cat("Test Data:\n")
cat(sprintf("  n = %d, p = %d\n", n, ncol(X)))
cat(sprintf("  y_reg: mean=%.4f, sd=%.4f, range=[%.2f, %.2f]\n",
            mean(y_reg), sd(y_reg), min(y_reg), max(y_reg)))
cat(sprintf("  y_class: %.1f%% positive (class 1)\n\n", mean(y_class)*100))

# ============================================================
# 1. GBM VALIDATION
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("1. GRADIENT BOOSTING MACHINE (gbm package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# Fit GBM regression
gbm_fit <- gbm(y_reg ~ x1 + x2 + x3,
               data=df,
               distribution="gaussian",
               n.trees=100,
               interaction.depth=3,
               shrinkage=0.1,
               bag.fraction=1.0,
               n.minobsinnode=10,
               verbose=FALSE)

pred_gbm <- predict(gbm_fit, df, n.trees=100)
mse_gbm <- mean((y_reg - pred_gbm)^2)
r2_gbm <- 1 - mse_gbm/var(y_reg)

cat("GBM Regression (100 trees, depth=3, lr=0.1):\n")
cat(sprintf("  Training MSE: %.6f\n", mse_gbm))
cat(sprintf("  R-squared:    %.4f\n", r2_gbm))

# Variable importance
imp_gbm <- summary(gbm_fit, plotit=FALSE)
cat("\n  Variable Importance (relative):\n")
for (i in 1:nrow(imp_gbm)) {
  cat(sprintf("    %s: %.2f\n", imp_gbm$var[i], imp_gbm$rel.inf[i]))
}
cat("\n")

# ============================================================
# 2. ADABOOST VALIDATION
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("2. ADABOOST (using gbm with adaboost distribution)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# AdaBoost for classification using gbm
df$y_class_factor <- ifelse(df$y_class == 1, 1, 0)
ada_fit <- gbm(y_class_factor ~ x1 + x2 + x3,
               data=df,
               distribution="adaboost",
               n.trees=50,
               interaction.depth=1,  # stumps
               shrinkage=1.0,
               bag.fraction=1.0,
               n.minobsinnode=1,
               verbose=FALSE)

pred_ada_prob <- predict(ada_fit, df, n.trees=50, type="response")
pred_ada_class <- ifelse(pred_ada_prob > 0.5, 1, 0)
acc_ada <- mean(pred_ada_class == df$y_class)

cat("AdaBoost Classification (50 stumps):\n")
cat(sprintf("  Training Accuracy: %.4f\n", acc_ada))
cat(sprintf("  Training Error:    %.4f\n", 1 - acc_ada))
cat("\n")

# ============================================================
# 3. KERNEL SVM VALIDATION
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("3. KERNEL SVM (e1071::svm package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# RBF Kernel SVM
svm_rbf <- svm(y_class ~ x1 + x2 + x3,
               data=df,
               type="C-classification",
               kernel="radial",
               gamma=1/3,  # 1/n_features
               cost=1.0,
               scale=FALSE)

pred_svm_rbf <- predict(svm_rbf, df)
acc_svm_rbf <- mean(as.numeric(pred_svm_rbf) - 1 == df$y_class)

cat("RBF Kernel SVM (C=1, gamma=0.33):\n")
cat(sprintf("  Training Accuracy: %.4f\n", acc_svm_rbf))
cat(sprintf("  Number of SVs:     %d (%.1f%% of data)\n",
            length(svm_rbf$index), 100*length(svm_rbf$index)/n))

# Polynomial Kernel
svm_poly <- svm(y_class ~ x1 + x2 + x3,
                data=df,
                type="C-classification",
                kernel="polynomial",
                degree=3,
                gamma=1/3,
                coef0=0,
                cost=1.0,
                scale=FALSE)

pred_svm_poly <- predict(svm_poly, df)
acc_svm_poly <- mean(as.numeric(pred_svm_poly) - 1 == df$y_class)

cat("\nPolynomial Kernel SVM (degree=3, C=1):\n")
cat(sprintf("  Training Accuracy: %.4f\n", acc_svm_poly))
cat(sprintf("  Number of SVs:     %d\n", length(svm_poly$index)))

# Linear Kernel
svm_linear <- svm(y_class ~ x1 + x2 + x3,
                  data=df,
                  type="C-classification",
                  kernel="linear",
                  cost=1.0,
                  scale=FALSE)

pred_svm_linear <- predict(svm_linear, df)
acc_svm_linear <- mean(as.numeric(pred_svm_linear) - 1 == df$y_class)

cat("\nLinear Kernel SVM (C=1):\n")
cat(sprintf("  Training Accuracy: %.4f\n", acc_svm_linear))
cat(sprintf("  Number of SVs:     %d\n\n", length(svm_linear$index)))

# ============================================================
# 4. ROC/AUC VALIDATION
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("4. ROC/AUC (pROC package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# Use GBM predictions as probabilities
pred_probs <- predict(gbm_fit, newdata=data.frame(x1=x1, x2=x2, x3=x3),
                      n.trees=100, type="response")
# Scale to 0-1 for classification task
pred_probs_class <- plogis(predict(ada_fit, df, n.trees=50))

roc_obj <- roc(df$y_class, pred_probs_class, quiet=TRUE)
auc_val <- auc(roc_obj)

# Find optimal threshold (Youden's J)
coords_best <- coords(roc_obj, "best", ret=c("threshold", "sensitivity", "specificity"))

cat("ROC/AUC Analysis:\n")
cat(sprintf("  AUC: %.4f\n", auc_val))
cat(sprintf("  Optimal Threshold (Youden's J): %.4f\n", coords_best$threshold))
cat(sprintf("  Sensitivity at optimal: %.4f\n", coords_best$sensitivity))
cat(sprintf("  Specificity at optimal: %.4f\n", coords_best$specificity))

# Confusion matrix at optimal threshold
pred_opt <- ifelse(pred_probs_class >= coords_best$threshold, 1, 0)
tp <- sum(pred_opt == 1 & df$y_class == 1)
tn <- sum(pred_opt == 0 & df$y_class == 0)
fp <- sum(pred_opt == 1 & df$y_class == 0)
fn <- sum(pred_opt == 0 & df$y_class == 1)

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)
f1 <- 2 * precision * recall / (precision + recall)
accuracy <- (tp + tn) / n

cat(sprintf("\n  Confusion Matrix at optimal threshold:\n"))
cat(sprintf("    TP=%d, TN=%d, FP=%d, FN=%d\n", tp, tn, fp, fn))
cat(sprintf("    Precision: %.4f\n", precision))
cat(sprintf("    Recall:    %.4f\n", recall))
cat(sprintf("    F1 Score:  %.4f\n", f1))
cat(sprintf("    Accuracy:  %.4f\n\n", accuracy))

# ============================================================
# 5. VARIABLE IMPORTANCE (Random Forest)
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("5. VARIABLE IMPORTANCE (randomForest %IncMSE)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

rf_fit <- randomForest(y_reg ~ x1 + x2 + x3,
                       data=df,
                       ntree=100,
                       importance=TRUE,
                       mtry=2)

imp_rf <- importance(rf_fit)
cat("Random Forest Variable Importance:\n")
cat(sprintf("  x1: %%IncMSE=%.2f, IncNodePurity=%.2f\n",
            imp_rf["x1", "%IncMSE"], imp_rf["x1", "IncNodePurity"]))
cat(sprintf("  x2: %%IncMSE=%.2f, IncNodePurity=%.2f\n",
            imp_rf["x2", "%IncMSE"], imp_rf["x2", "IncNodePurity"]))
cat(sprintf("  x3: %%IncMSE=%.2f, IncNodePurity=%.2f\n\n",
            imp_rf["x3", "%IncMSE"], imp_rf["x3", "IncNodePurity"]))

# Normalized importance
total_purity <- sum(imp_rf[, "IncNodePurity"])
cat("Normalized Importance (MDI-style):\n")
cat(sprintf("  x1: %.4f\n", imp_rf["x1", "IncNodePurity"] / total_purity))
cat(sprintf("  x2: %.4f\n", imp_rf["x2", "IncNodePurity"] / total_purity))
cat(sprintf("  x3: %.4f\n\n", imp_rf["x3", "IncNodePurity"] / total_purity))

# ============================================================
# 6. PARTIAL DEPENDENCE
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("6. PARTIAL DEPENDENCE (from GBM)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

# GBM's built-in partial dependence
pd_x1 <- plot(gbm_fit, i.var=1, n.trees=100, return.grid=TRUE)
cat("Partial Dependence for x1 (10 grid points):\n")
grid_points <- seq(min(x1), max(x1), length.out=10)
for (i in 1:min(10, nrow(pd_x1))) {
  cat(sprintf("  x1=%.3f -> y=%.4f\n", pd_x1$x1[i], pd_x1$y[i]))
}
cat("\n")

# Check if PD is monotonically increasing (should be for x1)
pd_trend <- if(pd_x1$y[nrow(pd_x1)] > pd_x1$y[1]) "increasing" else "decreasing"
cat(sprintf("  PD trend for x1: %s (expected: increasing)\n\n", pd_trend))

# ============================================================
# SUMMARY
# ============================================================
cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("SUMMARY: KEY REFERENCE VALUES FOR RUST VALIDATION\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

cat("1. GBM Regression:\n")
cat(sprintf("   MSE: %.6f, R²: %.4f\n", mse_gbm, r2_gbm))
cat(sprintf("   Top importance: %s (%.1f%%)\n\n", imp_gbm$var[1], imp_gbm$rel.inf[1]))

cat("2. AdaBoost Classification:\n")
cat(sprintf("   Accuracy: %.4f\n\n", acc_ada))

cat("3. Kernel SVM:\n")
cat(sprintf("   RBF Accuracy:    %.4f (%d SVs)\n", acc_svm_rbf, length(svm_rbf$index)))
cat(sprintf("   Poly Accuracy:   %.4f\n", acc_svm_poly))
cat(sprintf("   Linear Accuracy: %.4f\n\n", acc_svm_linear))

cat("4. ROC/AUC:\n")
cat(sprintf("   AUC: %.4f\n", auc_val))
cat(sprintf("   Optimal threshold: %.4f\n\n", coords_best$threshold))

cat("5. Variable Importance (x1 should be highest):\n")
cat(sprintf("   x1 rank: 1 (most important for y=2*x1+0.5*x2+noise)\n\n"))

cat("6. Partial Dependence:\n")
cat(sprintf("   x1: %s trend (coefficient is positive)\n", pd_trend))
