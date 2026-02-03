# Benchmark: Advanced Spatial Methods (sphet, splm, spatialprobit)
# Comparing R packages with Rust p2a-core implementations

library(spdep)
library(sphet)
library(splm)
library(spatialprobit)
library(Matrix)

set.seed(42)

# Helper function to time execution
benchmark_r <- function(expr, name, n_runs = 5) {
  times <- numeric(n_runs)
  for (i in 1:n_runs) {
    start <- Sys.time()
    result <- eval(expr)
    end <- Sys.time()
    times[i] <- as.numeric(difftime(end, start, units = "secs")) * 1000  # ms
  }
  cat(sprintf("%s: %.2f ms (median of %d runs)\n", name, median(times), n_runs))
  return(list(time_ms = median(times), result = result))
}

# Generate test data
generate_spatial_data <- function(n, rho = 0.4) {
  # Create grid coordinates
  side <- ceiling(sqrt(n))
  coords <- expand.grid(x = 1:side, y = 1:side)[1:n, ]

  # Create k-nearest neighbors
  knn <- knearneigh(as.matrix(coords), k = 4)
  nb <- knn2nb(knn)
  listw <- nb2listw(nb, style = "W")

  # Generate X
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  X <- cbind(1, x1, x2)

  # True parameters
  beta <- c(2, 0.5, -0.3)

  # Generate y with spatial lag
  W <- listw2mat(listw)
  A <- diag(n) - rho * W
  epsilon <- rnorm(n)
  y <- solve(A) %*% (X %*% beta + epsilon)

  data <- data.frame(y = as.vector(y), x1 = x1, x2 = x2)

  return(list(data = data, listw = listw, coords = coords, W = W))
}

generate_panel_data <- function(n_units, n_time, rho = 0.3) {
  # Create spatial structure for cross-section
  side <- ceiling(sqrt(n_units))
  coords <- expand.grid(x = 1:side, y = 1:side)[1:n_units, ]
  knn <- knearneigh(as.matrix(coords), k = 4)
  nb <- knn2nb(knn)
  listw <- nb2listw(nb, style = "W")

  # Generate panel data
  id <- rep(1:n_units, each = n_time)
  time <- rep(1:n_time, times = n_units)

  # Fixed effects
  alpha_i <- rnorm(n_units, sd = 1)

  # Covariates
  x1 <- rnorm(n_units * n_time)
  x2 <- rnorm(n_units * n_time)

  # Generate y (simplified, no spatial lag in DGP for speed)
  y <- 2 + 0.5 * x1 - 0.3 * x2 + alpha_i[id] + rnorm(n_units * n_time, sd = 0.5)

  data <- data.frame(id = id, time = time, y = y, x1 = x1, x2 = x2)

  return(list(data = data, listw = listw))
}

generate_binary_spatial_data <- function(n, rho = 0.3) {
  # Create grid coordinates
  side <- ceiling(sqrt(n))
  coords <- expand.grid(x = 1:side, y = 1:side)[1:n, ]

  # Create k-nearest neighbors
  knn <- knearneigh(as.matrix(coords), k = 4)
  nb <- knn2nb(knn)
  listw <- nb2listw(nb, style = "W")
  W <- listw2mat(listw)

  # Generate X
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  X <- cbind(x1, x2)

  # True parameters
  beta <- c(0.5, -0.3)

  # Generate latent y* with spatial lag
  A <- diag(n) - rho * W
  epsilon <- rnorm(n)
  y_star <- solve(A) %*% (X %*% beta + epsilon)

  # Binary outcome
  y <- as.integer(y_star > 0)

  data <- data.frame(y = y, x1 = x1, x2 = x2)

  return(list(data = data, listw = listw, W = W))
}

cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("BENCHMARK: Advanced Spatial Methods\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

# ============================================================================
# 1. SPHET BENCHMARKS
# ============================================================================
cat("1. SPHET (Spatial GMM with Heteroscedasticity)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

for (n in c(100, 400, 900)) {
  cat(sprintf("\nn = %d observations:\n", n))

  spatial_data <- generate_spatial_data(n, rho = 0.4)

  # SAR model via GMM
  tryCatch({
    res <- benchmark_r(
      quote(spreg(y ~ x1 + x2, data = spatial_data$data, listw = spatial_data$listw,
                  model = "lag", het = TRUE)),
      sprintf("  sphet SAR (n=%d)", n),
      n_runs = 3
    )
  }, error = function(e) {
    cat(sprintf("  sphet SAR (n=%d): ERROR - %s\n", n, e$message))
  })

  # SEM model via GMM
  tryCatch({
    res <- benchmark_r(
      quote(spreg(y ~ x1 + x2, data = spatial_data$data, listw = spatial_data$listw,
                  model = "error", het = TRUE)),
      sprintf("  sphet SEM (n=%d)", n),
      n_runs = 3
    )
  }, error = function(e) {
    cat(sprintf("  sphet SEM (n=%d): ERROR - %s\n", n, e$message))
  })
}

# ============================================================================
# 2. SPLM BENCHMARKS
# ============================================================================
cat("\n\n2. SPLM (Spatial Panel Models)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

for (config in list(c(25, 10), c(49, 10), c(100, 10))) {
  n_units <- config[1]
  n_time <- config[2]
  n_total <- n_units * n_time

  cat(sprintf("\nn = %d units x %d periods = %d observations:\n", n_units, n_time, n_total))

  panel_data <- generate_panel_data(n_units, n_time)

  # Spatial panel ML - fixed effects
  tryCatch({
    res <- benchmark_r(
      quote(spml(y ~ x1 + x2, data = panel_data$data, listw = panel_data$listw,
                 index = c("id", "time"), model = "within", effect = "individual",
                 lag = TRUE)),
      sprintf("  spml FE lag (n=%d)", n_total),
      n_runs = 3
    )
  }, error = function(e) {
    cat(sprintf("  spml FE lag (n=%d): ERROR - %s\n", n_total, e$message))
  })

  # Spatial panel GMM
  tryCatch({
    res <- benchmark_r(
      quote(spgm(y ~ x1 + x2, data = panel_data$data, listw = panel_data$listw,
                 index = c("id", "time"), model = "within", lag = TRUE)),
      sprintf("  spgm FE lag (n=%d)", n_total),
      n_runs = 3
    )
  }, error = function(e) {
    cat(sprintf("  spgm FE lag (n=%d): ERROR - %s\n", n_total, e$message))
  })
}

# ============================================================================
# 3. SPATIALPROBIT BENCHMARKS
# ============================================================================
cat("\n\n3. SPATIALPROBIT (Spatial Probit Models)\n")
cat("-" %>% rep(50) %>% paste(collapse = ""), "\n")

for (n in c(100, 225, 400)) {
  cat(sprintf("\nn = %d observations:\n", n))

  binary_data <- generate_binary_spatial_data(n, rho = 0.3)
  W_sparse <- as(binary_data$W, "CssparseMatrix")

  # SAR Probit (Bayesian MCMC) - use fewer iterations for benchmark
  tryCatch({
    res <- benchmark_r(
      quote(sarprobit(y ~ x1 + x2, W = W_sparse, data = binary_data$data,
                      ndraw = 500, burn.in = 100, showProgress = FALSE)),
      sprintf("  SAR probit (n=%d, 500 draws)", n),
      n_runs = 2  # Fewer runs since MCMC is slow
    )
  }, error = function(e) {
    cat(sprintf("  SAR probit (n=%d): ERROR - %s\n", n, e$message))
  })

  # SEM Probit
  tryCatch({
    res <- benchmark_r(
      quote(semprobit(y ~ x1 + x2, W = W_sparse, data = binary_data$data,
                      ndraw = 500, burn.in = 100, showProgress = FALSE)),
      sprintf("  SEM probit (n=%d, 500 draws)", n),
      n_runs = 2
    )
  }, error = function(e) {
    cat(sprintf("  SEM probit (n=%d): ERROR - %s\n", n, e$message))
  })
}

cat("\n\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
cat("BENCHMARK COMPLETE\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n")
