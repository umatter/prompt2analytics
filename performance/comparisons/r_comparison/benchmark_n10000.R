#!/usr/bin/env Rscript
# R Benchmarks at N=10,000 for Paper Main Text
# Runs all methods at consistent sample size
#
# Usage: Rscript benchmark_n10000.R

suppressPackageStartupMessages({
  library(bench)
  library(sandwich)
  library(plm)
  library(lfe)
  library(forecast)
})

set.seed(42)
N <- 10000

cat("=== R Benchmarks at N=10,000 ===\n\n")

# Data generators
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

# Benchmark runner
run_bench <- function(name, fn, iterations = 100) {
  result <- bench::mark(
    fn(),
    iterations = iterations,
    check = FALSE,
    memory = TRUE,
    filter_gc = FALSE
  )
  raw_times <- result$time[[1]]
  times_us <- as.numeric(raw_times) * 1e6
  mem_alloc <- as.numeric(result$mem_alloc[[1]])

  list(
    method = name,
    n = N,
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

format_bytes <- function(bytes) {
  if (is.na(bytes) || bytes == 0) return("0 B")
  if (abs(bytes) < 1024) return(sprintf("%d B", bytes))
  if (abs(bytes) < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  return(sprintf("%.2f MB", bytes / 1024^2))
}

print_result <- function(result) {
  cat(sprintf("%-12s  median: %10.1f µs  IQR: [%8.1f, %8.1f]  mem: %10s\n",
              result$method,
              result$time_median_us,
              result$time_p25_us,
              result$time_p75_us,
              format_bytes(result$mem_alloc_bytes)))
}

results <- list()
idx <- 1

# Regression
cat("--- Regression ---\n")
reg_data <- generate_regression_data(N)

r <- run_bench("OLS", function() lm(y ~ x1 + x2 + x3 + x4 + x5, data = reg_data))
results[[idx]] <- r; print_result(r); idx <- idx + 1

r <- run_bench("OLS+HC1", function() {
  fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = reg_data)
  vcovHC(fit, type = "HC1")
})
results[[idx]] <- r; print_result(r); idx <- idx + 1

# Panel (100 entities x 100 periods = 10,000)
cat("\n--- Panel Data ---\n")
panel_data <- generate_panel_data(100, 100)
pdata <- pdata.frame(panel_data, index = c("entity", "time"))

r <- run_bench("FE_plm", function() plm(y ~ x1 + x2, data = pdata, model = "within"))
results[[idx]] <- r; print_result(r); idx <- idx + 1

# Discrete Choice
cat("\n--- Discrete Choice ---\n")
binary_data <- generate_binary_data(N)

r <- run_bench("Logit", function() glm(y ~ x1 + x2, data = binary_data, family = binomial(link = "logit")))
results[[idx]] <- r; print_result(r); idx <- idx + 1

# Time Series (longer for N=10000)
cat("\n--- Time Series ---\n")
ts_data <- generate_time_series(N)

r <- run_bench("ARIMA", function() Arima(ts_data, order = c(1, 1, 1)), iterations = 20)
results[[idx]] <- r; print_result(r); idx <- idx + 1

r <- run_bench("MSTL", function() mstl(ts_data), iterations = 20)
results[[idx]] <- r; print_result(r); idx <- idx + 1

# ML
cat("\n--- Machine Learning ---\n")
set.seed(42)
k <- 5
ml_data <- matrix(0, nrow = N, ncol = k)
for (i in 1:N) {
  cluster <- (i - 1) %% 3
  center <- cluster * 3
  ml_data[i, ] <- center + runif(k, -0.5, 0.5)
}

r <- run_bench("K-Means", function() kmeans(ml_data, centers = 3, nstart = 5, iter.max = 100))
results[[idx]] <- r; print_result(r); idx <- idx + 1

r <- run_bench("PCA", function() prcomp(ml_data, center = TRUE, scale. = FALSE))
results[[idx]] <- r; print_result(r); idx <- idx + 1

# Save results
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

dir.create("results", showWarnings = FALSE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_file <- sprintf("results/r_n10000_%s.csv", timestamp)
write.csv(results_df, output_file, row.names = FALSE)

cat(sprintf("\n\nResults saved to: %s\n", output_file))
cat(sprintf("Total benchmarks: %d methods at N=%d\n", nrow(results_df), N))
