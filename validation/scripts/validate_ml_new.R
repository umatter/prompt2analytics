#!/usr/bin/env Rscript
# Validation script for newly implemented ML methods
# Compares: glmnet, rpart (CART)

library(glmnet)
library(rpart)

cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("ML Methods Validation: R Reference Values\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n\n")

# Test data - simple linear relationship
set.seed(42)
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)  # noise
y_reg <- 2 + 3*x1 - 1.5*x2 + rnorm(n, sd=0.5)

X <- cbind(x1, x2, x3)
colnames(X) <- c("x1", "x2", "x3")

# Binary classification data
y_class <- ifelse(x1 + x2 > 0, 1, 0)

cat("Test Data:\n")
cat(sprintf("  n = %d, p = %d\n", n, ncol(X)))
cat(sprintf("  y_reg: mean=%.4f, sd=%.4f\n", mean(y_reg), sd(y_reg)))
cat(sprintf("  y_class: %.1f%% positive\n\n", mean(y_class)*100))

# ============================================================
# 1. GLMNET Validation
# ============================================================
cat("-" |> rep(60) |> paste(collapse=""), "\n")
cat("1. GLMNET (Elastic Net / Lasso / Ridge)\n")
cat("-" |> rep(60) |> paste(collapse=""), "\n\n")

# Ridge (alpha=0)
ridge_fit <- glmnet(X, y_reg, alpha=0, lambda=0.1, standardize=TRUE)
cat("Ridge (alpha=0, lambda=0.1):\n")
cat(sprintf("  Intercept: %.6f\n", coef(ridge_fit)[1]))
cat(sprintf("  x1 coef:   %.6f\n", coef(ridge_fit)[2]))
cat(sprintf("  x2 coef:   %.6f\n", coef(ridge_fit)[3]))
cat(sprintf("  x3 coef:   %.6f\n\n", coef(ridge_fit)[4]))

# Lasso (alpha=1)
lasso_fit <- glmnet(X, y_reg, alpha=1, lambda=0.1, standardize=TRUE)
cat("Lasso (alpha=1, lambda=0.1):\n")
cat(sprintf("  Intercept: %.6f\n", coef(lasso_fit)[1]))
cat(sprintf("  x1 coef:   %.6f\n", coef(lasso_fit)[2]))
cat(sprintf("  x2 coef:   %.6f\n", coef(lasso_fit)[3]))
cat(sprintf("  x3 coef:   %.6f (should be ~0, noise)\n\n", coef(lasso_fit)[4]))

# Elastic Net (alpha=0.5)
enet_fit <- glmnet(X, y_reg, alpha=0.5, lambda=0.1, standardize=TRUE)
cat("Elastic Net (alpha=0.5, lambda=0.1):\n")
cat(sprintf("  Intercept: %.6f\n", coef(enet_fit)[1]))
cat(sprintf("  x1 coef:   %.6f\n", coef(enet_fit)[2]))
cat(sprintf("  x2 coef:   %.6f\n", coef(enet_fit)[3]))
cat(sprintf("  x3 coef:   %.6f\n\n", coef(enet_fit)[4]))

# Predictions
pred_ridge <- predict(ridge_fit, X)[,1]
pred_lasso <- predict(lasso_fit, X)[,1]
mse_ridge <- mean((y_reg - pred_ridge)^2)
mse_lasso <- mean((y_reg - pred_lasso)^2)
cat("Prediction MSE:\n")
cat(sprintf("  Ridge MSE: %.6f\n", mse_ridge))
cat(sprintf("  Lasso MSE: %.6f\n\n", mse_lasso))

# Cross-validation
set.seed(42)
cv_fit <- cv.glmnet(X, y_reg, alpha=1, nfolds=5)
cat("CV Lasso (5-fold):\n")
cat(sprintf("  lambda.min: %.6f\n", cv_fit$lambda.min))
cat(sprintf("  lambda.1se: %.6f\n", cv_fit$lambda.1se))
cat(sprintf("  CVM at min: %.6f\n\n", min(cv_fit$cvm)))

# ============================================================
# 2. RPART (CART) Validation
# ============================================================
cat("-" |> rep(60) |> paste(collapse=""), "\n")
cat("2. RPART (CART Decision Trees)\n")
cat("-" |> rep(60) |> paste(collapse=""), "\n\n")

df_reg <- data.frame(y=y_reg, x1=x1, x2=x2, x3=x3)

# Regression tree
rpart_reg <- rpart(y ~ x1 + x2 + x3,
                   data=df_reg,
                   method="anova",
                   control=rpart.control(maxdepth=5, minsplit=5, cp=0.01))

pred_rpart <- predict(rpart_reg, df_reg)
mse_rpart <- mean((y_reg - pred_rpart)^2)

cat("CART Regression (maxdepth=5, cp=0.01):\n")
cat(sprintf("  Training MSE: %.6f\n", mse_rpart))
cat(sprintf("  R-squared:    %.4f\n", 1 - mse_rpart/var(y_reg)))
cat(sprintf("  Num nodes:    %d\n", nrow(rpart_reg$frame)))
cat(sprintf("  Num leaves:   %d\n\n", sum(rpart_reg$frame$var == "<leaf>")))

# Variable importance
if (!is.null(rpart_reg$variable.importance)) {
  imp_rpart <- rpart_reg$variable.importance / sum(rpart_reg$variable.importance) * 100
  cat("Variable Importance:\n")
  for (nm in names(imp_rpart)) {
    cat(sprintf("  %s: %.2f%%\n", nm, imp_rpart[nm]))
  }
}
cat("\n")

# Classification tree
df_class <- data.frame(y=factor(y_class), x1=x1, x2=x2, x3=x3)
rpart_class <- rpart(y ~ x1 + x2 + x3,
                     data=df_class,
                     method="class",
                     control=rpart.control(maxdepth=5, minsplit=5, cp=0.01))

pred_class <- predict(rpart_class, df_class, type="class")
acc_rpart <- mean(pred_class == df_class$y)

cat("CART Classification (Gini):\n")
cat(sprintf("  Training Accuracy: %.4f\n", acc_rpart))
cat(sprintf("  Num nodes:         %d\n\n", nrow(rpart_class$frame)))

# ============================================================
# Summary
# ============================================================
cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("Key Reference Values for Validation\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n\n")

cat("GLMNET Ridge (λ=0.1):\n")
cat(sprintf("  Coefs: [%.4f, %.4f, %.4f, %.4f]\n",
    coef(ridge_fit)[1], coef(ridge_fit)[2], coef(ridge_fit)[3], coef(ridge_fit)[4]))

cat("\nGLMNET Lasso (λ=0.1):\n")
cat(sprintf("  Coefs: [%.4f, %.4f, %.4f, %.4f]\n",
    coef(lasso_fit)[1], coef(lasso_fit)[2], coef(lasso_fit)[3], coef(lasso_fit)[4]))

cat("\nCART Regression:\n")
cat(sprintf("  MSE: %.4f, R²: %.4f\n", mse_rpart, 1 - mse_rpart/var(y_reg)))

cat("\nCART Classification:\n")
cat(sprintf("  Accuracy: %.4f\n", acc_rpart))
