# Benchmark spatial econometrics: R vs Rust
# Compare: spdep (Moran's I, LM tests), spatialreg (lagsarlm, errorsarlm)

library(spdep)
library(spatialreg)
library(microbenchmark)

# Function to create grid coordinates and spatial weights
create_spatial_data <- function(n_side) {
  n <- n_side * n_side

  # Create grid coordinates
  coords <- expand.grid(x = 1:n_side, y = 1:n_side)
  coords <- as.matrix(coords)

  # Create k-nearest neighbors (k=4)
  nb <- knn2nb(knearneigh(coords, k = 4))
  listw <- nb2listw(nb, style = "W")

  # Generate spatially correlated data
  set.seed(42)
  x <- rnorm(n)

  # Create y with spatial lag: y = 0.5 * Wy + 2 + 0.7*x + error
  W <- listw2mat(listw)
  error <- rnorm(n, sd = 0.5)
  y <- solve(diag(n) - 0.5 * W) %*% (2 + 0.7 * x + error)
  y <- as.vector(y)

  data <- data.frame(y = y, x = x)

  list(data = data, coords = coords, nb = nb, listw = listw)
}

# Benchmark function
benchmark_spatial <- function(n_sides, n_iter = 10) {
  results <- list()

  for (n_side in n_sides) {
    n <- n_side * n_side
    cat(sprintf("\n=== Testing with n = %d (%dx%d grid) ===\n", n, n_side, n_side))

    # Create data
    spatial <- create_spatial_data(n_side)
    data <- spatial$data
    coords <- spatial$coords
    nb <- spatial$nb
    listw <- spatial$listw

    # Benchmark knearneigh + nb2listw (neighbor construction)
    neighbors_time <- microbenchmark(
      {
        nb_temp <- knn2nb(knearneigh(coords, k = 4))
        listw_temp <- nb2listw(nb_temp, style = "W")
      },
      times = n_iter
    )

    # Benchmark Moran's I
    moran_time <- microbenchmark(
      moran.test(data$y, listw, alternative = "greater"),
      times = n_iter
    )

    # Benchmark LM tests
    lm_model <- lm(y ~ x, data = data)
    lm_tests_time <- microbenchmark(
      lm.LMtests(lm_model, listw, test = c("LMlag", "LMerr", "RLMlag", "RLMerr", "SARMA")),
      times = n_iter
    )

    # Benchmark SAR model (lagsarlm)
    sar_time <- microbenchmark(
      lagsarlm(y ~ x, data = data, listw = listw),
      times = n_iter
    )

    # Benchmark SEM model (errorsarlm)
    sem_time <- microbenchmark(
      errorsarlm(y ~ x, data = data, listw = listw),
      times = n_iter
    )

    # Benchmark Local Moran's I (localmoran)
    localmoran_time <- microbenchmark(
      localmoran(data$y, listw),
      times = n_iter
    )

    # Benchmark SAC model (sacsarlm)
    sac_time <- microbenchmark(
      sacsarlm(y ~ x, data = data, listw = listw, type = "sac"),
      times = min(n_iter, 5)  # SAC is slow, fewer iterations
    )

    # Store results
    results[[as.character(n)]] <- list(
      n = n,
      neighbors_ms = median(neighbors_time$time) / 1e6,
      moran_ms = median(moran_time$time) / 1e6,
      lm_tests_ms = median(lm_tests_time$time) / 1e6,
      sar_ms = median(sar_time$time) / 1e6,
      sem_ms = median(sem_time$time) / 1e6,
      localmoran_ms = median(localmoran_time$time) / 1e6,
      sac_ms = median(sac_time$time) / 1e6
    )

    cat(sprintf("  neighbors:  %.2f ms\n", results[[as.character(n)]]$neighbors_ms))
    cat(sprintf("  moran.test: %.2f ms\n", results[[as.character(n)]]$moran_ms))
    cat(sprintf("  lm.LMtests: %.2f ms\n", results[[as.character(n)]]$lm_tests_ms))
    cat(sprintf("  lagsarlm:   %.2f ms\n", results[[as.character(n)]]$sar_ms))
    cat(sprintf("  errorsarlm: %.2f ms\n", results[[as.character(n)]]$sem_ms))
    cat(sprintf("  localmoran: %.2f ms\n", results[[as.character(n)]]$localmoran_ms))
    cat(sprintf("  sacsarlm:   %.2f ms\n", results[[as.character(n)]]$sac_ms))
  }

  # Create summary data frame
  df <- do.call(rbind, lapply(results, function(r) {
    data.frame(
      method = c("neighbors", "moran_test", "lm_tests", "lagsarlm", "errorsarlm",
                 "localmoran", "sacsarlm"),
      n = r$n,
      time_ms = c(r$neighbors_ms, r$moran_ms, r$lm_tests_ms, r$sar_ms, r$sem_ms,
                  r$localmoran_ms, r$sac_ms)
    )
  }))

  df
}

# Run benchmarks
cat("R Spatial Econometrics Benchmarks\n")
cat("==================================\n")
cat("Packages: spdep", packageVersion("spdep"), ", spatialreg", packageVersion("spatialreg"), "\n")

# Test with different grid sizes: 10x10=100, 32x32=1024
# Cap at 32x32: SAR/SEM require O(n^3) matrix operations, 100x100=10000 is impractical
results <- benchmark_spatial(c(10, 32), n_iter = 10)

# Save results
write.csv(results, "r_spatial_benchmark_results.csv", row.names = FALSE)

cat("\n\nResults saved to r_spatial_benchmark_results.csv\n")
print(results)
