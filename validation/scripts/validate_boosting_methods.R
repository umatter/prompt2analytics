#!/usr/bin/env Rscript
# Validation for XGBoost, LightGBM, MBoost, and BART
# Compares R reference implementations to Rust implementations

cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("BOOSTING METHODS VALIDATION: R Reference Values\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

# ============================================================
# Create test data (same as used in Rust tests)
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

X <- cbind(x1, x2, x3)

cat("Test Data:\n")
cat(sprintf("  n = %d, p = %d\n", n, ncol(X)))
cat(sprintf("  y_reg: mean=%.4f, sd=%.4f\n", mean(y_reg), sd(y_reg)))
cat(sprintf("  y_class: %.1f%% positive\n\n", mean(y_class)*100))

# ============================================================
# 1. XGBoost
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("1. XGBoost (xgboost package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

if (requireNamespace("xgboost", quietly = TRUE)) {
  library(xgboost)

  # Regression
  dtrain_reg <- xgb.DMatrix(X, label = y_reg)

  start_time <- Sys.time()
  xgb_reg <- xgb.train(
    params = list(
      objective = "reg:squarederror",
      eta = 0.3,
      max_depth = 6,
      lambda = 1.0,  # L2
      alpha = 0.0,   # L1
      subsample = 1.0,
      colsample_bytree = 1.0
    ),
    data = dtrain_reg,
    nrounds = 100,
    verbose = 0
  )
  xgb_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_xgb <- predict(xgb_reg, X)
  mse_xgb <- mean((y_reg - pred_xgb)^2)
  r2_xgb <- 1 - mse_xgb / var(y_reg)

  cat("XGBoost Regression (eta=0.3, depth=6, 100 rounds):\n")
  cat(sprintf("  MSE: %.6f\n", mse_xgb))
  cat(sprintf("  R²:  %.4f\n", r2_xgb))
  cat(sprintf("  Time: %.4f seconds\n\n", xgb_time))

  # Feature importance
  imp_xgb <- xgb.importance(model = xgb_reg)
  cat("Feature Importance (Gain):\n")
  for (i in 1:nrow(imp_xgb)) {
    cat(sprintf("  %s: %.4f\n", imp_xgb$Feature[i], imp_xgb$Gain[i]))
  }
  cat("\n")

  # Classification
  dtrain_class <- xgb.DMatrix(X, label = y_class)

  start_time <- Sys.time()
  xgb_class <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eta = 0.3,
      max_depth = 6,
      lambda = 1.0
    ),
    data = dtrain_class,
    nrounds = 100,
    verbose = 0
  )
  xgb_class_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_xgb_class <- predict(xgb_class, X)
  pred_labels <- ifelse(pred_xgb_class > 0.5, 1, 0)
  acc_xgb <- mean(pred_labels == y_class)

  cat("XGBoost Classification:\n")
  cat(sprintf("  Accuracy: %.4f\n", acc_xgb))
  cat(sprintf("  Time: %.4f seconds\n\n", xgb_class_time))

} else {
  cat("xgboost package not installed. Install with: install.packages('xgboost')\n\n")
}

# ============================================================
# 2. LightGBM
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("2. LightGBM (lightgbm package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

if (requireNamespace("lightgbm", quietly = TRUE)) {
  library(lightgbm)

  # Regression
  dtrain_lgb <- lgb.Dataset(X, label = y_reg)

  start_time <- Sys.time()
  lgb_reg <- lgb.train(
    params = list(
      objective = "regression",
      learning_rate = 0.1,
      num_leaves = 31,
      max_bin = 255,
      min_data_in_leaf = 20,
      lambda_l1 = 0.0,
      lambda_l2 = 0.0
    ),
    data = dtrain_lgb,
    nrounds = 100,
    verbose = -1
  )
  lgb_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_lgb <- predict(lgb_reg, X)
  mse_lgb <- mean((y_reg - pred_lgb)^2)
  r2_lgb <- 1 - mse_lgb / var(y_reg)

  cat("LightGBM Regression (lr=0.1, leaves=31, 100 rounds):\n")
  cat(sprintf("  MSE: %.6f\n", mse_lgb))
  cat(sprintf("  R²:  %.4f\n", r2_lgb))
  cat(sprintf("  Time: %.4f seconds\n\n", lgb_time))

  # Feature importance
  imp_lgb <- lgb.importance(lgb_reg)
  cat("Feature Importance (Gain):\n")
  for (i in 1:nrow(imp_lgb)) {
    cat(sprintf("  %s: %.4f\n", imp_lgb$Feature[i], imp_lgb$Gain[i]))
  }
  cat("\n")

  # Classification
  dtrain_lgb_class <- lgb.Dataset(X, label = y_class)

  start_time <- Sys.time()
  lgb_class <- lgb.train(
    params = list(
      objective = "binary",
      learning_rate = 0.1,
      num_leaves = 31
    ),
    data = dtrain_lgb_class,
    nrounds = 100,
    verbose = -1
  )
  lgb_class_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_lgb_class <- predict(lgb_class, X)
  pred_labels_lgb <- ifelse(pred_lgb_class > 0.5, 1, 0)
  acc_lgb <- mean(pred_labels_lgb == y_class)

  cat("LightGBM Classification:\n")
  cat(sprintf("  Accuracy: %.4f\n", acc_lgb))
  cat(sprintf("  Time: %.4f seconds\n\n", lgb_class_time))

} else {
  cat("lightgbm package not installed. Install with: install.packages('lightgbm')\n\n")
}

# ============================================================
# 3. MBoost
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("3. MBoost (mboost package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

if (requireNamespace("mboost", quietly = TRUE)) {
  library(mboost)

  df <- data.frame(y = y_reg, x1 = x1, x2 = x2, x3 = x3)

  # Tree base learner (btree)
  start_time <- Sys.time()
  mb_tree <- mboost(
    y ~ btree(x1) + btree(x2) + btree(x3),
    data = df,
    control = boost_control(mstop = 100, nu = 0.1)
  )
  mb_tree_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_mb_tree <- predict(mb_tree, df)
  mse_mb_tree <- mean((y_reg - pred_mb_tree)^2)
  r2_mb_tree <- 1 - mse_mb_tree / var(y_reg)

  cat("MBoost with Tree base learner (nu=0.1, mstop=100):\n")
  cat(sprintf("  MSE: %.6f\n", mse_mb_tree))
  cat(sprintf("  R²:  %.4f\n", r2_mb_tree))
  cat(sprintf("  Time: %.4f seconds\n\n", mb_tree_time))

  # Componentwise linear (bols)
  start_time <- Sys.time()
  mb_linear <- mboost(
    y ~ bols(x1) + bols(x2) + bols(x3),
    data = df,
    control = boost_control(mstop = 100, nu = 0.1)
  )
  mb_linear_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_mb_linear <- predict(mb_linear, df)
  mse_mb_linear <- mean((y_reg - pred_mb_linear)^2)
  r2_mb_linear <- 1 - mse_mb_linear / var(y_reg)

  cat("MBoost with Linear base learner (componentwise):\n")
  cat(sprintf("  MSE: %.6f\n", mse_mb_linear))
  cat(sprintf("  R²:  %.4f\n", r2_mb_linear))
  cat(sprintf("  Time: %.4f seconds\n\n", mb_linear_time))

  # Variable selection frequency
  cat("Variable Selection (from linear model):\n")
  sel_freq <- varimp(mb_linear)
  for (nm in names(sel_freq)) {
    cat(sprintf("  %s: %.4f\n", nm, sel_freq[nm]))
  }
  cat("\n")

} else {
  cat("mboost package not installed. Install with: install.packages('mboost')\n\n")
}

# ============================================================
# 4. BART
# ============================================================
cat("-" |> rep(70) |> paste(collapse=""), "\n")
cat("4. BART (BART package)\n")
cat("-" |> rep(70) |> paste(collapse=""), "\n\n")

if (requireNamespace("BART", quietly = TRUE)) {
  library(BART)

  # Regression BART
  start_time <- Sys.time()
  bart_reg <- wbart(
    x.train = X,
    y.train = y_reg,
    ntree = 50,
    nskip = 100,
    ndpost = 200,
    printevery = 0
  )
  bart_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  pred_bart <- bart_reg$yhat.train.mean
  mse_bart <- mean((y_reg - pred_bart)^2)
  r2_bart <- 1 - mse_bart / var(y_reg)

  cat("BART Regression (ntree=50, burn=100, post=200):\n")
  cat(sprintf("  MSE: %.6f\n", mse_bart))
  cat(sprintf("  R²:  %.4f\n", r2_bart))
  cat(sprintf("  Sigma: %.4f (posterior mean)\n", mean(bart_reg$sigma)))
  cat(sprintf("  Time: %.4f seconds\n\n", bart_time))

  # Variable importance (count of times each variable was used)
  cat("Variable Importance (usage counts):\n")
  if (!is.null(bart_reg$varcount)) {
    varcount <- colMeans(bart_reg$varcount)
    names(varcount) <- paste0("x", 1:ncol(X))
    varcount_norm <- varcount / sum(varcount)
    for (i in 1:length(varcount_norm)) {
      cat(sprintf("  x%d: %.4f\n", i, varcount_norm[i]))
    }
  }
  cat("\n")

  # Prediction intervals
  cat("Prediction Intervals (95% credible):\n")
  lower <- apply(bart_reg$yhat.train, 2, quantile, 0.025)
  upper <- apply(bart_reg$yhat.train, 2, quantile, 0.975)
  coverage <- mean(y_reg >= lower & y_reg <= upper)
  cat(sprintf("  Coverage: %.1f%% (target: 95%%)\n", coverage * 100))
  cat(sprintf("  Mean width: %.4f\n\n", mean(upper - lower)))

} else {
  cat("BART package not installed. Install with: install.packages('BART')\n\n")
}

# ============================================================
# BENCHMARK SUMMARY
# ============================================================
cat("=" |> rep(70) |> paste(collapse=""), "\n")
cat("BENCHMARK SUMMARY\n")
cat("=" |> rep(70) |> paste(collapse=""), "\n\n")

cat("Method              | R² (Regression) | Time (s) | Notes\n")
cat("--------------------|-----------------|----------|------\n")

if (exists("r2_xgb")) {
  cat(sprintf("XGBoost             | %.4f          | %.4f   | eta=0.3, depth=6\n", r2_xgb, xgb_time))
}
if (exists("r2_lgb")) {
  cat(sprintf("LightGBM            | %.4f          | %.4f   | lr=0.1, leaves=31\n", r2_lgb, lgb_time))
}
if (exists("r2_mb_tree")) {
  cat(sprintf("MBoost (tree)       | %.4f          | %.4f   | nu=0.1, mstop=100\n", r2_mb_tree, mb_tree_time))
}
if (exists("r2_mb_linear")) {
  cat(sprintf("MBoost (linear)     | %.4f          | %.4f   | componentwise\n", r2_mb_linear, mb_linear_time))
}
if (exists("r2_bart")) {
  cat(sprintf("BART                | %.4f          | %.4f   | ntree=50\n", r2_bart, bart_time))
}

cat("\n")
cat("Expected R² for y = 2*x1 + 0.5*x2 + noise should be >0.90\n")
cat("x1 should have highest importance (coef=2 vs x2 coef=0.5)\n")
