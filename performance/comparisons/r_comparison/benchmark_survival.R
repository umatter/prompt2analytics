#!/usr/bin/env Rscript
# Survival Analysis Benchmarks: R survival package
# Compares R survival package against p2a Rust implementation
#
# Install required packages:
# install.packages(c("bench", "survival"))

suppressPackageStartupMessages({
  library(bench)
  library(survival)
})

set.seed(42)

cat("=== R Survival Analysis Benchmarks ===\n\n")

# ============================================
# Data Generators (matching Rust DGP)
# ============================================

generate_survival_data <- function(n, censoring_rate = 0.3) {
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  group <- rep(c("A", "B"), length.out = n)

  # Generate true event time using Weibull distribution
  linear <- 0.5 * x1 + 0.3 * x2 + ifelse(group == "B", 0.5, 0)
  u <- runif(n, 0.0001, 0.9999)
  shape <- 1.5
  scale <- 10.0
  true_time <- scale * (-log(u))^(1/shape) * exp(-linear)

  # Generate censoring time (exponential)
  censor_rate_adj <- censoring_rate * 0.1 * scale
  censor_time <- -censor_rate_adj * log(runif(n, 0.0001, 0.9999))

  # Observed = min(event, censor)
  time <- pmin(true_time, censor_time)
  event <- as.integer(true_time < censor_time)

  data.frame(time = time, event = event, x1 = x1, x2 = x2, group = group)
}

generate_competing_risks_data <- function(n) {
  # Generate times for each event type
  time1 <- -10.0 * log(runif(n, 0.0001, 0.9999))  # Event type 1
  time2 <- -8.0 * log(runif(n, 0.0001, 0.9999))   # Event type 2
  censor <- -15.0 * log(runif(n, 0.0001, 0.9999)) # Censoring

  # Find which occurs first
  time <- pmin(time1, time2, censor)
  event_type <- ifelse(time1 < time2 & time1 < censor, 1,
                       ifelse(time2 < time1 & time2 < censor, 2, 0))

  data.frame(time = time, event_type = event_type)
}

# ============================================
# Benchmark Runner
# ============================================

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

format_bytes <- function(bytes) {
  if (is.na(bytes) || bytes == 0) return("0 B")
  if (abs(bytes) < 1024) return(sprintf("%d B", bytes))
  if (abs(bytes) < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  return(sprintf("%.2f MB", bytes / 1024^2))
}

print_result <- function(result) {
  cat(sprintf("%-30s n=%6d  median: %12.1f us  IQR: [%10.1f, %10.1f]  mem: %10s\n",
              result$method,
              result$n,
              result$time_median_us,
              result$time_p25_us,
              result$time_p75_us,
              format_bytes(result$mem_alloc_bytes)))
}

# ============================================
# Run Benchmarks
# ============================================

results <- list()
idx <- 1

# --- Kaplan-Meier ---
cat("\n--- Kaplan-Meier ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_survival_data(n)

  # Unstratified
  r <- run_bench(sprintf("KM_unstratified_%d", n),
                 function() survfit(Surv(time, event) ~ 1, data = data))
  r$n <- n
  r$method <- "KM_unstratified"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # Stratified by group
  r <- run_bench(sprintf("KM_stratified_%d", n),
                 function() survfit(Surv(time, event) ~ group, data = data))
  r$n <- n
  r$method <- "KM_stratified"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Log-Rank Test ---
cat("\n--- Log-Rank Test ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_survival_data(n)

  r <- run_bench(sprintf("LogRank_%d", n),
                 function() survdiff(Surv(time, event) ~ group, data = data))
  r$n <- n
  r$method <- "LogRank"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Cox Proportional Hazards ---
cat("\n--- Cox PH ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_survival_data(n)

  # Efron method (default in R)
  r <- run_bench(sprintf("CoxPH_efron_%d", n),
                 function() coxph(Surv(time, event) ~ x1 + x2, data = data, ties = "efron"))
  r$n <- n
  r$method <- "CoxPH_efron"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # Breslow method
  r <- run_bench(sprintf("CoxPH_breslow_%d", n),
                 function() coxph(Surv(time, event) ~ x1 + x2, data = data, ties = "breslow"))
  r$n <- n
  r$method <- "CoxPH_breslow"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- AFT Models ---
cat("\n--- AFT Models ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_survival_data(n)

  # Weibull AFT
  r <- run_bench(sprintf("AFT_weibull_%d", n),
                 function() survreg(Surv(time, event) ~ x1 + x2, data = data, dist = "weibull"),
                 iterations = 50)
  r$n <- n
  r$method <- "AFT_weibull"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1

  # Log-Normal AFT
  r <- run_bench(sprintf("AFT_lognormal_%d", n),
                 function() survreg(Surv(time, event) ~ x1 + x2, data = data, dist = "lognormal"),
                 iterations = 50)
  r$n <- n
  r$method <- "AFT_lognormal"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# --- Competing Risks (Aalen-Johansen) ---
cat("\n--- Competing Risks ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_competing_risks_data(n)

  # Use survfit with multi-state for Aalen-Johansen
  r <- run_bench(sprintf("CompetingRisks_%d", n),
                 function() survfit(Surv(time, factor(event_type)) ~ 1, data = data))
  r$n <- n
  r$method <- "CompetingRisks"
  results[[idx]] <- r
  print_result(r)
  idx <- idx + 1
}

# ============================================
# Save Results
# ============================================

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
output_file <- sprintf("results/survival_benchmarks_%s.csv", timestamp)
write.csv(results_df, output_file, row.names = FALSE)

cat(sprintf("\n\nResults saved to: %s\n", output_file))

# ============================================
# Print Summary Table
# ============================================

cat("\n=== Summary Table (median times in microseconds) ===\n\n")

# Aggregate by method and n
summary_table <- aggregate(time_median_us ~ method + n, data = results_df, FUN = median)
summary_table <- summary_table[order(summary_table$method, summary_table$n), ]

# Print as comparison table
cat("| Method              | n=100    | n=500    | n=1000   | n=2000/5000 |\n")
cat("|---------------------|----------|----------|----------|-------------|\n")

methods <- c("KM_unstratified", "KM_stratified", "LogRank",
             "CoxPH_efron", "CoxPH_breslow",
             "AFT_weibull", "AFT_lognormal", "CompetingRisks")

for (m in methods) {
  row <- summary_table[summary_table$method == m, ]
  vals <- rep(NA, 4)

  for (i in 1:nrow(row)) {
    n_val <- row$n[i]
    time_val <- row$time_median_us[i]

    if (n_val == 100) vals[1] <- time_val
    else if (n_val == 500) vals[2] <- time_val
    else if (n_val == 1000) vals[3] <- time_val
    else vals[4] <- time_val
  }

  cat(sprintf("| %-19s | %8.1f | %8.1f | %8.1f | %11.1f |\n",
              m,
              ifelse(is.na(vals[1]), 0, vals[1]),
              ifelse(is.na(vals[2]), 0, vals[2]),
              ifelse(is.na(vals[3]), 0, vals[3]),
              ifelse(is.na(vals[4]), 0, vals[4])))
}

cat("\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("survival package version: %s\n", packageVersion("survival")))
