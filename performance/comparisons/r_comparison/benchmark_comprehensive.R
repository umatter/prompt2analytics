#!/usr/bin/env Rscript
# Comprehensive R Benchmarks with Distribution Statistics
# Uses the 'bench' package for detailed timing and memory statistics
# Output format matches p2a Rust comprehensive benchmarks
#
# Install required packages:
# install.packages(c("bench", "sandwich", "plm", "lfe", "forecast", "changepoint"))

suppressPackageStartupMessages({
  library(bench)
  library(sandwich)
  library(plm)
  library(lfe)
})

# Ensure reproducibility
set.seed(42)

cat("=== R Comprehensive Benchmarks (using bench package) ===\n\n")

# ============================================
# Data Generators (matching Rust DGP)
# ============================================

generate_regression_data <- function(n, k = 5) {
  X <- matrix(runif(n * k, -1, 1), nrow = n, ncol = k)
  colnames(X) <- paste0("x", 1:k)
  y <- rowSums(X) + runif(n, 0, 0.5)
  data.frame(y = y, X)
}

generate_panel_data <- function(n_entities, n_periods) {
  n <- n_entities * n_periods
  entity <- rep(1:n_entities, each = n_periods)
  time <- rep(1:n_periods, times = n_entities)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  entity_effect <- (entity - 1) * 0.1
  y <- entity_effect + 0.5 * x1 + 0.3 * x2 + runif(n, 0, 0.5)
  data.frame(entity = entity, time = time, y = y, x1 = x1, x2 = x2)
}

generate_binary_data <- function(n) {
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  linear <- -1 + 0.5 * x1 + 0.3 * x2
  prob <- 1 / (1 + exp(-linear))
  y <- as.numeric(runif(n) < prob)
  data.frame(y = y, x1 = x1, x2 = x2)
}

generate_time_series <- function(n) {
  t <- 1:n
  trend <- 0.01 * t
  seasonal <- sin(t * pi / 6) * 2.0
  noise <- runif(n, 0, 0.5)
  ts(trend + seasonal + noise, frequency = 12)
}

# ============================================
# Benchmark Runner
# ============================================

run_bench <- function(name, fn, iterations = 100) {
  # Use a function wrapper to ensure proper evaluation
  # bench::mark expects expressions, but functions work more reliably
  result <- bench::mark(
    fn(),
    iterations = iterations,
    check = FALSE,
    memory = TRUE,
    filter_gc = FALSE
  )

  # Extract timing distribution
  # bench_time objects store values in seconds
  raw_times <- result$time[[1]]
  times_us <- as.numeric(raw_times) * 1e6  # Convert seconds to microseconds

  # Memory in bytes
  mem_alloc <- as.numeric(result$mem_alloc[[1]])

  list(
    method = name,
    n = NA,
    iterations = length(times_us),
    time_min_us = min(times_us),
    time_p25_us = quantile(times_us, 0.25),
    time_median_us = median(times_us),
    time_p75_us = quantile(times_us, 0.75),
    time_max_us = max(times_us),
    time_mean_us = mean(times_us),
    time_std_us = sd(times_us),
    itr_per_sec = 1e6 / median(times_us),
    mem_alloc_bytes = mem_alloc
  )
}

# Format bytes for display
format_bytes <- function(bytes) {
  if (is.na(bytes) || bytes == 0) return("0 B")
  if (abs(bytes) < 1024) return(sprintf("%d B", bytes))
  if (abs(bytes) < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  return(sprintf("%.2f MB", bytes / 1024^2))
}

# Print result in consistent format
print_result <- function(result) {
  cat(sprintf("%-20s n=%6d  median: %10.1f µs  IQR: [%8.1f, %8.1f]  itr/s: %8.1f  mem: %10s\n",
              result$method,
              result$n,
              result$time_median_us,
              result$time_p25_us,
              result$time_p75_us,
              result$itr_per_sec,
              format_bytes(result$mem_alloc_bytes)))
}

# ============================================
# Run Benchmarks
# ============================================

results <- list()
idx <- 1

# --- Regression ---
cat("\n--- Regression ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_regression_data(n)

  # OLS Standard
  r <- run_bench(sprintf("OLS_standard_%d", n), function() lm(y ~ x1 + x2 + x3 + x4 + x5, data = data))
  r$n <- n
  r$method <- "OLS"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # OLS + HC1
  r <- run_bench(sprintf("OLS_HC1_%d", n), function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = data)
    vcovHC(fit, type = "HC1")
  })
  r$n <- n
  r$method <- "OLS+HC1"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Panel Data ---
cat("\n--- Panel Data ---\n")

for (params in list(c(10, 10), c(50, 20), c(100, 100))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  data <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(data, index = c("entity", "time"))

  # Fixed Effects (plm)
  r <- run_bench(sprintf("FE_plm_%d", n), function() plm(y ~ x1 + x2, data = pdata, model = "within"))
  r$n <- n
  r$method <- "FE_plm"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # Fixed Effects (lfe)
  r <- run_bench(sprintf("FE_lfe_%d", n), function() felm(y ~ x1 + x2 | entity, data = data))
  r$n <- n
  r$method <- "FE_lfe"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # HDFE (lfe)
  r <- run_bench(sprintf("HDFE_%d", n), function() felm(y ~ x1 + x2 | entity + time, data = data))
  r$n <- n
  r$method <- "HDFE"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Discrete Choice ---
cat("\n--- Discrete Choice ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_binary_data(n)

  # Logit
  r <- run_bench(sprintf("Logit_%d", n),
                 function() glm(y ~ x1 + x2, data = data, family = binomial(link = "logit")))
  r$n <- n
  r$method <- "Logit"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # Probit
  r <- run_bench(sprintf("Probit_%d", n),
                 function() glm(y ~ x1 + x2, data = data, family = binomial(link = "probit")))
  r$n <- n
  r$method <- "Probit"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Time Series ---
cat("\n--- Time Series ---\n")

if (require(forecast, quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    ts_data <- generate_time_series(n)

    # ARIMA
    r <- run_bench(sprintf("ARIMA_%d", n), function() Arima(ts_data, order = c(1, 1, 1)), iterations = 20)
    r$n <- n
    r$method <- "ARIMA"
    results[[idx]] <- r
    print_result(r)
    idx <- idx + 1

    # MSTL
    r <- run_bench(sprintf("MSTL_%d", n), function() mstl(ts_data), iterations = 20)
    r$n <- n
    r$method <- "MSTL"
    results[[idx]] <- r
    print_result(r)
    idx <- idx + 1
  }
}

# --- ML ---
cat("\n--- Machine Learning ---\n")

for (n in c(100, 1000, 10000)) {
  # Generate cluster data
  set.seed(42)
  k <- 5
  ml_data <- matrix(0, nrow = n, ncol = k)
  for (i in 1:n) {
    cluster <- (i - 1) %% 3
    center <- cluster * 3
    ml_data[i, ] <- center + runif(k, -0.5, 0.5)
  }

  # K-Means
  r <- run_bench(sprintf("KMeans_%d", n),
                 function() kmeans(ml_data, centers = 3, nstart = 5, iter.max = 100))
  r$n <- n
  r$method <- "K-Means"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # PCA
  r <- run_bench(sprintf("PCA_%d", n), function() prcomp(ml_data, center = TRUE, scale. = FALSE))
  r$n <- n
  r$method <- "PCA"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# ============================================
# Save Results
# ============================================

# Convert to data frame
results_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    method = r$method,
    n = r$n,
    iterations = r$iterations,
    time_min_us = r$time_min_us,
    time_p25_us = r$time_p25_us,
    time_median_us = r$time_median_us,
    time_p75_us = r$time_p75_us,
    time_max_us = r$time_max_us,
    time_mean_us = r$time_mean_us,
    time_std_us = r$time_std_us,
    itr_per_sec = r$itr_per_sec,
    mem_alloc_bytes = r$mem_alloc_bytes
  )
}))

# Save to CSV
dir.create("results", showWarnings = FALSE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_file <- sprintf("results/r_comprehensive_%s.csv", timestamp)
write.csv(results_df, output_file, row.names = FALSE)

cat(sprintf("\n\nResults saved to: %s\n", output_file))

# Print summary
cat("\n=== Summary ===\n")
cat(sprintf("Total benchmarks: %d\n", nrow(results_df)))
cat(sprintf("R version: %s\n", R.version.string))

# Print distribution example
if (nrow(results_df) > 0) {
  r <- results_df[1, ]
  cat(sprintf("\nExample distribution (%s n=%d):\n", r$method, r$n))
  cat(sprintf("  Min:    %10.1f µs\n", r$time_min_us))
  cat(sprintf("  P25:    %10.1f µs\n", r$time_p25_us))
  cat(sprintf("  Median: %10.1f µs\n", r$time_median_us))
  cat(sprintf("  P75:    %10.1f µs\n", r$time_p75_us))
  cat(sprintf("  Max:    %10.1f µs\n", r$time_max_us))
  cat(sprintf("  Mean:   %10.1f µs\n", r$time_mean_us))
  cat(sprintf("  Std:    %10.1f µs\n", r$time_std_us))
  cat(sprintf("  Memory: %s\n", format_bytes(r$mem_alloc_bytes)))
}
