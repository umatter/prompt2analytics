#!/usr/bin/env Rscript
# Benchmark spatial methods: Local Moran's I and SAC model
# Compare against Rust implementation

library(spdep)
library(spatialreg)

# Create grid coordinates
create_grid_coords <- function(n_side) {
  coords <- expand.grid(x = 0:(n_side-1), y = 0:(n_side-1))
  as.matrix(coords)
}

# Create spatial data with patterns
create_spatial_data <- function(n_side, seed = 42) {
  set.seed(seed)
  n <- n_side^2

  # Create coordinates
  coords <- create_grid_coords(n_side)

  # Create k-nearest neighbors (k=4)
  nb <- knn2nb(knearneigh(coords, k = 4))
  listw <- nb2listw(nb, style = "W")

  # Generate x variable
  x <- runif(n, -1, 1)

  # Generate y with spatial pattern
  y <- 2 + 0.7 * x + 0.3 * (coords[,1] + coords[,2]) / n_side + rnorm(n, 0, 0.25)

  list(coords = coords, x = x, y = y, nb = nb, listw = listw, n = n)
}

# Benchmark function with timing
benchmark <- function(expr, n_iter = 10) {
  times <- numeric(n_iter)
  for (i in 1:n_iter) {
    start <- Sys.time()
    result <- eval(expr)
    end <- Sys.time()
    times[i] <- as.numeric(end - start, units = "secs") * 1000  # Convert to ms
  }
  list(mean_ms = mean(times), sd_ms = sd(times), min_ms = min(times), max_ms = max(times))
}

cat("=" , rep("-", 60), "=\n", sep="")
cat("R Spatial Methods Benchmark (Local Moran's I and SAC)\n")
cat("=" , rep("-", 60), "=\n", sep="")

results <- data.frame(
  method = character(),
  n = integer(),
  time_ms = numeric(),
  notes = character(),
  stringsAsFactors = FALSE
)

# Benchmark Local Moran's I
cat("\n--- Local Moran's I (localmoran) ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    localmoran(data$y, data$listw)
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
cat("\n--- Local Moran's I with permutation (99 perm) ---\n")
for (n_side in c(10, 20)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)

  timing <- benchmark(quote({
    localmoran_perm(data$y, data$listw, nsim = 99)
  }), n_iter = 5)  # Fewer iterations for permutation

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "localmoran_perm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "99 permutations"
  ))
}

# Benchmark SAC model (sacsarlm)
cat("\n--- SAC Model (sacsarlm) ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)
  df <- data.frame(y = data$y, x = data$x)

  timing <- benchmark(quote({
    sacsarlm(y ~ x, data = df, listw = data$listw, type = "sac")
  }), n_iter = 5)  # SAC is slow, fewer iterations

  cat(sprintf("n=%d: %.3f ms (±%.3f ms)\n", n, timing$mean_ms, timing$sd_ms))
  results <- rbind(results, data.frame(
    method = "sacsarlm",
    n = n,
    time_ms = timing$mean_ms,
    notes = "SARAR model"
  ))
}

# Compare with previously benchmarked methods for reference
cat("\n--- Reference: SAR and SEM for comparison ---\n")
for (n_side in c(10, 20, 32)) {
  n <- n_side^2
  data <- create_spatial_data(n_side)
  df <- data.frame(y = data$y, x = data$x)

  # SAR
  sar_timing <- benchmark(quote({
    lagsarlm(y ~ x, data = df, listw = data$listw)
  }), n_iter = 5)

  # SEM
  sem_timing <- benchmark(quote({
    errorsarlm(y ~ x, data = df, listw = data$listw)
  }), n_iter = 5)

  cat(sprintf("n=%d: SAR=%.3f ms, SEM=%.3f ms\n", n, sar_timing$mean_ms, sem_timing$mean_ms))

  results <- rbind(results, data.frame(
    method = "lagsarlm",
    n = n,
    time_ms = sar_timing$mean_ms,
    notes = "spatial lag"
  ))
  results <- rbind(results, data.frame(
    method = "errorsarlm",
    n = n,
    time_ms = sem_timing$mean_ms,
    notes = "spatial error"
  ))
}

# Write results to CSV
write.csv(results, "r_spatial_new_benchmark_results.csv", row.names = FALSE)

cat("\n", "=" , rep("-", 60), "=\n", sep="")
cat("Results saved to r_spatial_new_benchmark_results.csv\n")
cat("=" , rep("-", 60), "=\n", sep="")
