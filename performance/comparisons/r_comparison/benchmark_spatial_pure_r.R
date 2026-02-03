#!/usr/bin/env Rscript
# Pure R implementation of spatial methods for benchmarking
# No external dependencies required

# Create grid coordinates
create_grid_coords <- function(n_side) {
  coords <- expand.grid(x = 0:(n_side-1), y = 0:(n_side-1))
  as.matrix(coords)
}

# Create k-nearest neighbors
knn_neighbors <- function(coords, k) {
  n <- nrow(coords)
  neighbors <- vector("list", n)

  for (i in 1:n) {
    # Calculate distances to all other points
    distances <- sqrt(rowSums((coords - matrix(coords[i,], n, 2, byrow=TRUE))^2))
    distances[i] <- Inf  # Exclude self

    # Get k nearest neighbors
    neighbors[[i]] <- order(distances)[1:k]
  }

  neighbors
}

# Create row-standardized weights from neighbors
create_listw <- function(neighbors) {
  n <- length(neighbors)
  weights <- vector("list", n)

  for (i in 1:n) {
    ni <- length(neighbors[[i]])
    weights[[i]] <- rep(1/ni, ni)
  }

  list(neighbours = neighbors, weights = weights)
}

# Apply spatial lag W*y
spatial_lag <- function(y, listw) {
  n <- length(y)
  result <- numeric(n)

  for (i in 1:n) {
    nbs <- listw$neighbours[[i]]
    wts <- listw$weights[[i]]
    result[i] <- sum(wts * y[nbs])
  }

  result
}

# Compute eigenvalues of sparse W (for comparison with Rust)
compute_eigenvalues <- function(listw) {
  n <- length(listw$neighbours)

  # Convert to dense matrix
  W <- matrix(0, n, n)
  for (i in 1:n) {
    nbs <- listw$neighbours[[i]]
    wts <- listw$weights[[i]]
    W[i, nbs] <- wts
  }

  eigen(W)$values
}

# Local Moran's I
local_moran <- function(x, listw) {
  n <- length(x)
  z <- x - mean(x)
  s2 <- sum(z^2) / n

  local_i <- numeric(n)
  wx <- spatial_lag(z, listw)

  for (i in 1:n) {
    local_i[i] <- (z[i] / s2) * wx[i]
  }

  list(local_i = local_i, n = n)
}

# Local Moran's I with permutation
local_moran_perm <- function(x, listw, nsim = 99) {
  n <- length(x)
  z <- x - mean(x)
  s2 <- sum(z^2) / n

  # Observed local I
  wx <- spatial_lag(z, listw)
  local_i <- (z / s2) * wx

  # Permutation null distributions
  local_i_sim <- matrix(0, n, nsim)

  for (sim in 1:nsim) {
    z_perm <- sample(z)
    wx_perm <- spatial_lag(z_perm, listw)
    local_i_sim[, sim] <- (z / s2) * wx_perm  # Note: z stays fixed, wx uses permuted neighbors
  }

  # Compute p-values
  p_values <- numeric(n)
  for (i in 1:n) {
    # Two-sided test
    extreme_count <- sum(abs(local_i_sim[i, ]) >= abs(local_i[i]))
    p_values[i] <- (extreme_count + 1) / (nsim + 1)
  }

  list(local_i = local_i, p_values = p_values, n = n)
}

# SAC model (simplified implementation - 2D optimization)
sac_ml <- function(y, x, listw, tol = 1e-6, max_iter = 100, grid_res = 10) {
  n <- length(y)
  k <- ncol(x)

  # Get eigenvalues for log-determinant
  eigenvalues <- compute_eigenvalues(listw)
  real_eig <- Re(eigenvalues)

  param_min <- max(1/min(real_eig), -0.99)
  param_max <- min(1/max(real_eig), 0.99)

  # Pre-compute Wy and WX
  Wy <- spatial_lag(y, listw)
  WWy <- spatial_lag(Wy, listw)
  WX <- matrix(0, n, k)
  for (j in 1:k) {
    WX[,j] <- spatial_lag(x[,j], listw)
  }

  # Concentrated log-likelihood
  neg_ll <- function(rho, lambda) {
    if (rho <= param_min || rho >= param_max || lambda <= param_min || lambda >= param_max) {
      return(Inf)
    }

    # y* = (I - lambda*W)(I - rho*W)y
    wy_rho <- Wy - rho * WWy
    y_star <- y - rho * Wy - lambda * wy_rho

    # X* = (I - lambda*W)X
    X_star <- x - lambda * WX

    # beta = (X*'X*)^{-1} X*'y*
    XtX <- crossprod(X_star)
    Xty <- crossprod(X_star, y_star)
    beta <- solve(XtX, Xty)

    # sigma^2
    resid <- y_star - X_star %*% beta
    rss <- sum(resid^2)
    sigma2 <- rss / n

    if (sigma2 <= 0) return(Inf)

    # Log determinants
    log_det_rho <- sum(log(1 - rho * real_eig))
    log_det_lambda <- sum(log(1 - lambda * real_eig))

    if (!is.finite(log_det_rho) || !is.finite(log_det_lambda)) return(Inf)

    0.5 * n * (1 + log(2*pi) + log(sigma2)) - log_det_rho - log_det_lambda
  }

  # Grid search for initial values
  best_rho <- 0
  best_lambda <- 0
  best_ll <- Inf

  step <- (param_max - param_min) / grid_res
  for (i in 1:(grid_res-1)) {
    rho <- param_min + i * step
    for (j in 1:(grid_res-1)) {
      lambda <- param_min + j * step
      ll <- neg_ll(rho, lambda)
      if (ll < best_ll) {
        best_ll <- ll
        best_rho <- rho
        best_lambda <- lambda
      }
    }
  }

  # Coordinate descent refinement
  phi <- (1 + sqrt(5)) / 2

  for (iter in 1:max_iter) {
    old_rho <- best_rho
    old_lambda <- best_lambda

    # Optimize rho with lambda fixed
    a <- param_min
    b <- param_max
    c <- b - (b - a) / phi
    d <- a + (b - a) / phi

    for (inner in 1:50) {
      if (abs(b - a) < tol) break
      fc <- neg_ll(c, best_lambda)
      fd <- neg_ll(d, best_lambda)
      if (fc < fd) {
        b <- d
        d <- c
        c <- b - (b - a) / phi
      } else {
        a <- c
        c <- d
        d <- a + (b - a) / phi
      }
    }
    best_rho <- (a + b) / 2

    # Optimize lambda with rho fixed
    a <- param_min
    b <- param_max
    c <- b - (b - a) / phi
    d <- a + (b - a) / phi

    for (inner in 1:50) {
      if (abs(b - a) < tol) break
      fc <- neg_ll(best_rho, c)
      fd <- neg_ll(best_rho, d)
      if (fc < fd) {
        b <- d
        d <- c
        c <- b - (b - a) / phi
      } else {
        a <- c
        c <- d
        d <- a + (b - a) / phi
      }
    }
    best_lambda <- (a + b) / 2

    if (abs(best_rho - old_rho) < tol && abs(best_lambda - old_lambda) < tol) {
      break
    }
  }

  list(rho = best_rho, lambda = best_lambda, log_lik = -neg_ll(best_rho, best_lambda))
}

# SAR model
sar_ml <- function(y, x, listw, tol = 1e-8, max_iter = 100) {
  n <- length(y)
  k <- ncol(x)

  eigenvalues <- compute_eigenvalues(listw)
  real_eig <- Re(eigenvalues)

  rho_min <- max(1/min(real_eig), -0.99)
  rho_max <- min(1/max(real_eig), 0.99)

  Wy <- spatial_lag(y, listw)
  XtX_inv <- solve(crossprod(x))

  neg_ll <- function(rho) {
    y_tilde <- y - rho * Wy
    beta <- XtX_inv %*% crossprod(x, y_tilde)
    resid <- y_tilde - x %*% beta
    rss <- sum(resid^2)
    sigma2 <- rss / n
    log_det <- sum(log(1 - rho * real_eig))
    0.5 * n * log(sigma2) - log_det
  }

  # Golden section search
  phi <- (1 + sqrt(5)) / 2
  a <- rho_min
  b <- rho_max
  c <- b - (b - a) / phi
  d <- a + (b - a) / phi

  for (iter in 1:max_iter) {
    if (abs(b - a) < tol) break
    fc <- neg_ll(c)
    fd <- neg_ll(d)
    if (fc < fd) {
      b <- d
      d <- c
      c <- b - (b - a) / phi
    } else {
      a <- c
      c <- d
      d <- a + (b - a) / phi
    }
  }

  rho_opt <- (a + b) / 2
  list(rho = rho_opt, log_lik = -neg_ll(rho_opt))
}

# SEM model
sem_ml <- function(y, x, listw, tol = 1e-8, max_iter = 100) {
  n <- length(y)
  k <- ncol(x)

  eigenvalues <- compute_eigenvalues(listw)
  real_eig <- Re(eigenvalues)

  lambda_min <- max(1/min(real_eig), -0.99)
  lambda_max <- min(1/max(real_eig), 0.99)

  Wy <- spatial_lag(y, listw)
  WX <- matrix(0, n, k)
  for (j in 1:k) {
    WX[,j] <- spatial_lag(x[,j], listw)
  }

  neg_ll <- function(lambda) {
    y_star <- y - lambda * Wy
    X_star <- x - lambda * WX
    XtX_inv <- solve(crossprod(X_star))
    beta <- XtX_inv %*% crossprod(X_star, y_star)
    resid <- y_star - X_star %*% beta
    rss <- sum(resid^2)
    sigma2 <- rss / n
    log_det <- sum(log(1 - lambda * real_eig))
    0.5 * n * log(sigma2) - log_det
  }

  phi <- (1 + sqrt(5)) / 2
  a <- lambda_min
  b <- lambda_max
  c <- b - (b - a) / phi
  d <- a + (b - a) / phi

  for (iter in 1:max_iter) {
    if (abs(b - a) < tol) break
    fc <- neg_ll(c)
    fd <- neg_ll(d)
    if (fc < fd) {
      b <- d
      d <- c
      c <- b - (b - a) / phi
    } else {
      a <- c
      c <- d
      d <- a + (b - a) / phi
    }
  }

  lambda_opt <- (a + b) / 2
  list(lambda = lambda_opt, log_lik = -neg_ll(lambda_opt))
}

# Create spatial data
create_spatial_data <- function(n_side, seed = 42) {
  set.seed(seed)
  n <- n_side^2

  coords <- create_grid_coords(n_side)
  neighbors <- knn_neighbors(coords, 4)
  listw <- create_listw(neighbors)

  x_var <- runif(n, -1, 1)
  y_var <- 2 + 0.7 * x_var + 0.3 * (coords[,1] + coords[,2]) / n_side + rnorm(n, 0, 0.25)

  X <- cbind(1, x_var)  # Design matrix with intercept

  list(y = y_var, x = X, listw = listw, n = n)
}

# Benchmark function
benchmark <- function(expr, n_iter = 10) {
  times <- numeric(n_iter)
  for (i in 1:n_iter) {
    start <- Sys.time()
    result <- eval(expr)
    end <- Sys.time()
    times[i] <- as.numeric(end - start, units = "secs") * 1000
  }
  list(mean_ms = mean(times), sd_ms = sd(times), min_ms = min(times))
}

cat("=", rep("-", 60), "=\n", sep="")
cat("Pure R Spatial Methods Benchmark\n")
cat("=", rep("-", 60), "=\n\n", sep="")

results <- data.frame(
  method = character(),
  n = integer(),
  time_ms = numeric(),
  notes = character(),
  stringsAsFactors = FALSE
)

# Benchmark Local Moran's I
cat("--- Local Moran's I (analytical) ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    local_moran(data$y, data$listw)
  }))

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "localmoran",
    n = n,
    time_ms = timing$mean_ms,
    notes = "analytical"
  ))
}

# Benchmark Local Moran's I with permutation
cat("\n--- Local Moran's I (99 permutations) ---\n")
for (n_side in c(10, 20)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    local_moran_perm(data$y, data$listw, nsim = 99)
  }), n_iter = 5)

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "localmoran_perm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "99 permutations"
  ))
}

# Benchmark SAC model
cat("\n--- SAC Model ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    sac_ml(data$y, data$x, data$listw)
  }), n_iter = 5)

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "sacsarlm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "SARAR model"
  ))
}

# Benchmark SAR and SEM for reference
cat("\n--- SAR Model (for reference) ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    sar_ml(data$y, data$x, data$listw)
  }), n_iter = 5)

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "lagsarlm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "spatial lag"
  ))
}

cat("\n--- SEM Model (for reference) ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    sem_ml(data$y, data$x, data$listw)
  }), n_iter = 5)

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "errorsarlm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "spatial error"
  ))
}

# Write results
write.csv(results, "r_spatial_pure_benchmark_results.csv", row.names = FALSE)

cat("\n=", rep("-", 60), "=\n", sep="")
cat("Results saved to r_spatial_pure_benchmark_results.csv\n")
cat("=", rep("-", 60), "=\n", sep="")
