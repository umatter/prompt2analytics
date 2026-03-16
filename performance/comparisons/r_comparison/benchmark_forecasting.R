#!/usr/bin/env Rscript
# Forecasting Benchmarks for Cross-Language Comparison
# Compares R forecast package against p2a Rust implementation

library(microbenchmark)

# Check for forecast package
if (!require(forecast, quietly = TRUE)) {
  cat("forecast package required. Install with: install.packages('forecast')\n")
  quit(status = 1)
}

# Ensure reproducibility
set.seed(42)

# Generate time series with trend and seasonality
generate_time_series <- function(n) {
  t <- 1:n
  trend <- 0.01 * t
  seasonal <- sin(t * pi / 6) * 2.0
  noise <- runif(n, 0, 0.5)

  ts(trend + seasonal + noise, frequency = 12)
}

# Generate time series with changepoints
generate_changepoint_series <- function(n, n_changes = 3) {
  segment_size <- n %/% (n_changes + 1)

  series <- numeric(n)
  level <- 0

  for (i in 1:n) {
    if (i > 1 && (i - 1) %% segment_size == 0) {
      level <- level + runif(1, -2.5, 2.5)  # Random level shift
    }
    series[i] <- level + runif(1, 0, 0.3)
  }

  ts(series)
}

# Benchmark ARIMA
benchmark_arima <- function() {
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("Benchmarking ARIMA with n=%d\n", n))
    data <- generate_time_series(n)

    # ARIMA(1,1,1)
    bm <- microbenchmark(
      Arima(data, order = c(1, 1, 1)),
      times = 20,
      unit = "microseconds"
    )

    results[[paste0("arima_", n)]] <- summary(bm)
  }

  results
}

# Benchmark MSTL
benchmark_mstl <- function() {
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("Benchmarking MSTL with n=%d\n", n))
    data <- generate_time_series(n)

    bm <- microbenchmark(
      mstl(data),
      times = 20,
      unit = "microseconds"
    )

    results[[paste0("mstl_", n)]] <- summary(bm)
  }

  results
}

# Benchmark Changepoint Detection
benchmark_changepoint <- function() {
  # Check if changepoint package is available
  if (!require(changepoint, quietly = TRUE)) {
    cat("changepoint package not installed, skipping\n")
    return(list())
  }

  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("Benchmarking Changepoint with n=%d\n", n))
    data <- generate_changepoint_series(n, n_changes = 3)

    bm <- microbenchmark(
      cpt.mean(data, penalty = "BIC", method = "BinSeg", Q = 5),
      times = 20,
      unit = "microseconds"
    )

    results[[paste0("changepoint_", n)]] <- summary(bm)
  }

  results
}

# Run benchmarks
cat("=== ARIMA Benchmarks ===\n")
arima_results <- benchmark_arima()

cat("\n=== MSTL Benchmarks ===\n")
mstl_results <- benchmark_mstl()

cat("\n=== Changepoint Benchmarks ===\n")
changepoint_results <- benchmark_changepoint()

# Save results
save_results <- function(results, filename) {
  if (length(results) == 0) {
    cat(sprintf("No results to save for %s\n", filename))
    return()
  }

  df <- do.call(rbind, lapply(names(results), function(name) {
    r <- results[[name]]
    data.frame(
      method = name,
      mean_us = r$mean,
      median_us = r$median,
      min_us = r$min,
      max_us = r$max,
      n_eval = r$neval
    )
  }))

  write.csv(df, filename, row.names = FALSE)
  cat(sprintf("Results saved to %s\n", filename))
}

# Create results directory if needed
dir.create("results", showWarnings = FALSE)

save_results(arima_results, "results/forecasting_arima.csv")
save_results(mstl_results, "results/forecasting_mstl.csv")
save_results(changepoint_results, "results/forecasting_changepoint.csv")

# Print summary
cat("\n=== Summary ===\n")
cat("ARIMA:\n")
print(do.call(rbind, arima_results))
cat("\nMSTL:\n")
print(do.call(rbind, mstl_results))
cat("\nChangepoint:\n")
if (length(changepoint_results) > 0) print(do.call(rbind, changepoint_results))
