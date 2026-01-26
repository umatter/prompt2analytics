# Simplified spatial benchmark without sf dependency
# Uses spdep's basic matrix operations

# Try to load packages, install if needed
if (!require("microbenchmark", quietly = TRUE)) {
  install.packages("microbenchmark", repos = "https://cloud.r-project.org", quiet = TRUE)
}

library(microbenchmark)

# Manual implementation of spatial functions for benchmarking
# This mirrors what spdep does internally

# Create k-nearest neighbors from coordinates
create_knn_neighbors <- function(coords, k) {
  n <- nrow(coords)
  neighbors <- vector("list", n)

  for (i in 1:n) {
    # Calculate distances to all other points
    dists <- sqrt((coords[, 1] - coords[i, 1])^2 + (coords[, 2] - coords[i, 2])^2)
    dists[i] <- Inf  # Exclude self
    # Get k nearest neighbors
    neighbors[[i]] <- order(dists)[1:k]
  }

  neighbors
}

# Create row-standardized weights matrix
create_weights_matrix <- function(neighbors, n) {
  W <- matrix(0, n, n)
  for (i in 1:n) {
    nb <- neighbors[[i]]
    W[i, nb] <- 1 / length(nb)
  }
  W
}

# Moran's I statistic
moran_i <- function(y, W) {
  n <- length(y)
  y_centered <- y - mean(y)

  # Numerator: sum of y_i * W_ij * y_j
  num <- sum(y_centered * (W %*% y_centered))

  # Denominator: sum of (y - mean)^2
  denom <- sum(y_centered^2)

  # S0: sum of all weights
  S0 <- sum(W)

  I <- (n / S0) * (num / denom)

  # Expected value under null
  E_I <- -1 / (n - 1)

  # Variance (simplified)
  S1 <- 0.5 * sum((W + t(W))^2)
  S2 <- sum((rowSums(W) + colSums(W))^2)

  k <- (sum(y_centered^4) / n) / (sum(y_centered^2) / n)^2

  Var_I <- (n * ((n^2 - 3*n + 3) * S1 - n * S2 + 3 * S0^2) -
            k * ((n^2 - n) * S1 - 2 * n * S2 + 6 * S0^2)) /
           ((n - 1) * (n - 2) * (n - 3) * S0^2) - E_I^2

  z <- (I - E_I) / sqrt(Var_I)
  p_value <- 2 * (1 - pnorm(abs(z)))

  list(statistic = I, expectation = E_I, variance = Var_I, z = z, p_value = p_value)
}

# LM test for spatial lag
lm_lag_test <- function(residuals, X, W) {
  n <- length(residuals)
  e <- residuals

  # Wy
  Wy <- W %*% e

  # Numerator
  num <- (t(e) %*% Wy)^2

  # Trace terms for denominator
  WtW <- t(W) %*% W
  WpWt <- W + t(W)

  # M = I - X(X'X)^{-1}X'
  XtX_inv <- solve(t(X) %*% X)
  M <- diag(n) - X %*% XtX_inv %*% t(X)

  # T = tr(WMW'M + W'WM)
  WM <- W %*% M
  sigma2 <- sum(e^2) / n

  T_stat <- sum(diag(WM %*% t(WM))) + sum(diag(t(W) %*% W %*% M))

  LM <- as.numeric(num / (sigma2^2 * T_stat))
  p_value <- 1 - pchisq(LM, 1)

  list(statistic = LM, p_value = p_value)
}

# SAR model estimation (ML)
sar_ml <- function(y, X, W, tol = 1e-8, max_iter = 100) {
  n <- length(y)
  k <- ncol(X)

  # Eigenvalues of W for log determinant
  eig_W <- eigen(W, only.values = TRUE)$values
  rho_min <- 1 / min(Re(eig_W))
  rho_max <- 1 / max(Re(eig_W))

  # Log determinant function
  log_det <- function(rho) {
    sum(log(abs(1 - rho * Re(eig_W))))
  }

  # Concentrated log-likelihood
  neg_ll <- function(rho) {
    A <- diag(n) - rho * W
    Ay <- A %*% y

    # OLS on transformed data
    beta <- solve(t(X) %*% X) %*% t(X) %*% Ay
    resid <- Ay - X %*% beta
    sigma2 <- sum(resid^2) / n

    # Negative log-likelihood
    ll <- log_det(rho) - (n/2) * log(sigma2)
    -ll
  }

  # Golden section search
  result <- optimize(neg_ll, interval = c(rho_min + 0.01, rho_max - 0.01), tol = tol)
  rho_opt <- result$minimum

  # Final estimates
  A <- diag(n) - rho_opt * W
  Ay <- A %*% y
  beta <- solve(t(X) %*% X) %*% t(X) %*% Ay
  resid <- Ay - X %*% beta
  sigma2 <- sum(resid^2) / n

  list(rho = rho_opt, coefficients = as.vector(beta), sigma2 = sigma2,
       log_lik = -result$objective)
}

# SEM model estimation (ML)
sem_ml <- function(y, X, W, tol = 1e-8, max_iter = 100) {
  n <- length(y)
  k <- ncol(X)

  # Eigenvalues of W for log determinant
  eig_W <- eigen(W, only.values = TRUE)$values
  lambda_min <- 1 / min(Re(eig_W))
  lambda_max <- 1 / max(Re(eig_W))

  # Log determinant function
  log_det <- function(lambda) {
    sum(log(abs(1 - lambda * Re(eig_W))))
  }

  # Concentrated log-likelihood
  neg_ll <- function(lambda) {
    B <- diag(n) - lambda * W

    # Transform y and X
    By <- B %*% y
    BX <- B %*% X

    # OLS on transformed data
    beta <- solve(t(BX) %*% BX) %*% t(BX) %*% By
    resid <- By - BX %*% beta
    sigma2 <- sum(resid^2) / n

    # Negative log-likelihood
    ll <- log_det(lambda) - (n/2) * log(sigma2)
    -ll
  }

  # Golden section search
  result <- optimize(neg_ll, interval = c(lambda_min + 0.01, lambda_max - 0.01), tol = tol)
  lambda_opt <- result$minimum

  # Final estimates
  B <- diag(n) - lambda_opt * W
  By <- B %*% y
  BX <- B %*% X
  beta <- solve(t(BX) %*% BX) %*% t(BX) %*% By
  resid <- By - BX %*% beta
  sigma2 <- sum(resid^2) / n

  list(lambda = lambda_opt, coefficients = as.vector(beta), sigma2 = sigma2,
       log_lik = -result$objective)
}

# Create spatial data
create_spatial_data <- function(n_side, seed = 42) {
  set.seed(seed)
  n <- n_side * n_side

  # Grid coordinates
  coords <- expand.grid(x = 1:n_side, y = 1:n_side)
  coords <- as.matrix(coords)

  # K-nearest neighbors
  neighbors <- create_knn_neighbors(coords, k = 4)
  W <- create_weights_matrix(neighbors, n)

  # Generate x
  x <- rnorm(n)

  # Generate y with spatial pattern
  error <- rnorm(n, sd = 0.5)
  y <- 2 + 0.7 * x + 0.3 * (coords[, 1] + coords[, 2]) / n_side + error

  X <- cbind(1, x)  # Design matrix with intercept

  list(coords = coords, neighbors = neighbors, W = W, y = y, x = x, X = X, n = n)
}

# Benchmark function
benchmark_spatial <- function(n_sides, n_iter = 10) {
  results <- data.frame()

  for (n_side in n_sides) {
    n <- n_side * n_side
    cat(sprintf("\n=== Testing with n = %d (%dx%d grid) ===\n", n, n_side, n_side))

    # Pre-create data
    data <- create_spatial_data(n_side)

    # Benchmark neighbors + weights
    neighbors_time <- microbenchmark(
      {
        nb <- create_knn_neighbors(data$coords, k = 4)
        W <- create_weights_matrix(nb, data$n)
      },
      times = n_iter
    )

    # Benchmark Moran's I
    moran_time <- microbenchmark(
      moran_i(data$y, data$W),
      times = n_iter
    )

    # Benchmark LM tests
    # First fit OLS
    ols <- lm(data$y ~ data$x)
    residuals <- ols$residuals

    lm_tests_time <- microbenchmark(
      lm_lag_test(residuals, data$X, data$W),
      times = n_iter
    )

    # Benchmark SAR
    sar_time <- microbenchmark(
      sar_ml(data$y, data$X, data$W),
      times = n_iter
    )

    # Benchmark SEM
    sem_time <- microbenchmark(
      sem_ml(data$y, data$X, data$W),
      times = n_iter
    )

    cat(sprintf("  neighbors:  %.2f ms\n", median(neighbors_time$time) / 1e6))
    cat(sprintf("  moran_test: %.2f ms\n", median(moran_time$time) / 1e6))
    cat(sprintf("  lm_tests:   %.2f ms\n", median(lm_tests_time$time) / 1e6))
    cat(sprintf("  lagsarlm:   %.2f ms\n", median(sar_time$time) / 1e6))
    cat(sprintf("  errorsarlm: %.2f ms\n", median(sem_time$time) / 1e6))

    results <- rbind(results, data.frame(
      method = c("neighbors", "moran_test", "lm_tests", "lagsarlm", "errorsarlm"),
      n = n,
      time_ms = c(
        median(neighbors_time$time) / 1e6,
        median(moran_time$time) / 1e6,
        median(lm_tests_time$time) / 1e6,
        median(sar_time$time) / 1e6,
        median(sem_time$time) / 1e6
      )
    ))
  }

  results
}

# Run benchmarks
cat("R Spatial Econometrics Benchmarks (Pure R Implementation)\n")
cat("=========================================================\n")
cat("This mirrors spdep/spatialreg functionality in pure R\n\n")

results <- benchmark_spatial(c(10, 20, 32), n_iter = 10)

# Save results
write.csv(results, "r_spatial_benchmark_results.csv", row.names = FALSE)

cat("\n\nResults saved to r_spatial_benchmark_results.csv\n")
print(results)
