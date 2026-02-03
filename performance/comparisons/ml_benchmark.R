#!/usr/bin/env Rscript
# Performance benchmark: R implementations of ML methods

library(glmnet)
library(rpart)
library(microbenchmark)

cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("ML Methods Performance Benchmark: R\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n\n")

# Generate benchmark data
set.seed(42)
sizes <- c(1000, 5000, 10000, 50000)

for (n in sizes) {
  cat(sprintf("\n--- n = %d observations ---\n", n))

  p <- 10  # features
  X <- matrix(rnorm(n * p), n, p)
  y_reg <- X[,1] * 3 + X[,2] * (-1.5) + rnorm(n, sd=0.5)
  y_class <- factor(ifelse(X[,1] + X[,2] > 0, 1, 0))

  # 1. GLMNET Ridge
  cat("\nglmnet Ridge (alpha=0, single lambda):\n")
  time_ridge <- system.time({
    for (i in 1:10) {
      fit <- glmnet(X, y_reg, alpha=0, lambda=0.1)
    }
  })
  cat(sprintf("  Total (10 runs): %.3f sec\n", time_ridge["elapsed"]))
  cat(sprintf("  Per run:         %.4f sec\n", time_ridge["elapsed"]/10))

  # 2. GLMNET Lasso with CV
  if (n <= 10000) {  # CV is slow for large n
    cat("\nglmnet Lasso CV (5-fold):\n")
    time_cv <- system.time({
      fit <- cv.glmnet(X, y_reg, alpha=1, nfolds=5)
    })
    cat(sprintf("  Time: %.3f sec\n", time_cv["elapsed"]))
  }

  # 3. GLMNET path (100 lambdas)
  cat("\nglmnet path (100 lambdas):\n")
  time_path <- system.time({
    for (i in 1:5) {
      fit <- glmnet(X, y_reg, alpha=0.5, nlambda=100)
    }
  })
  cat(sprintf("  Total (5 runs): %.3f sec\n", time_path["elapsed"]))
  cat(sprintf("  Per run:        %.4f sec\n", time_path["elapsed"]/5))

  # 4. RPART Regression
  df <- data.frame(y=y_reg, X)
  cat("\nrpart regression (depth=5):\n")
  time_rpart <- system.time({
    for (i in 1:10) {
      fit <- rpart(y ~ ., data=df, control=rpart.control(maxdepth=5))
    }
  })
  cat(sprintf("  Total (10 runs): %.3f sec\n", time_rpart["elapsed"]))
  cat(sprintf("  Per run:         %.4f sec\n", time_rpart["elapsed"]/10))

  # 5. RPART Classification
  df_class <- data.frame(y=y_class, X)
  cat("\nrpart classification (depth=5):\n")
  time_rpart_class <- system.time({
    for (i in 1:10) {
      fit <- rpart(y ~ ., data=df_class, method="class",
                   control=rpart.control(maxdepth=5))
    }
  })
  cat(sprintf("  Total (10 runs): %.3f sec\n", time_rpart_class["elapsed"]))
  cat(sprintf("  Per run:         %.4f sec\n", time_rpart_class["elapsed"]/10))
}

cat("\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n")
cat("Benchmark complete\n")
cat("=" |> rep(60) |> paste(collapse=""), "\n")
