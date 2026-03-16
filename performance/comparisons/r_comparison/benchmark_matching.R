#!/usr/bin/env Rscript
# Propensity Score Matching R Benchmark (MatchIt, WeightIt)

library(bench)
set.seed(42)

run_bench <- function(name, fn, iterations = 100) {
  result <- bench::mark(fn(), iterations = iterations, check = FALSE, memory = TRUE, filter_gc = FALSE)
  raw_times <- result$time[[1]]
  times_us <- as.numeric(raw_times) * 1e6
  mem_alloc <- as.numeric(result$mem_alloc[[1]])
  list(method = name, n = NA, iterations = length(times_us),
    time_min_us = min(times_us), time_p25_us = quantile(times_us, 0.25),
    time_median_us = median(times_us), time_p75_us = quantile(times_us, 0.75),
    time_max_us = max(times_us), time_mean_us = mean(times_us),
    time_std_us = sd(times_us), itr_per_sec = 1e6 / median(times_us),
    mem_alloc_bytes = mem_alloc)
}

generate_treatment_data <- function(n, seed = 42) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
  ps <- plogis(0.5 + 0.3*x1 + 0.2*x2 - 0.1*x3)
  treatment <- rbinom(n, 1, ps)
  y <- 2 + 0.5*treatment + 0.3*x1 + 0.2*x2 + 0.1*x3 + rnorm(n)
  data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2, x3 = x3)
}

cat("=== Matching R Benchmarks (bench::mark) ===\n\n")

results <- list()
idx <- 1

# MatchIt
if (requireNamespace("MatchIt", quietly = TRUE)) {
  library(MatchIt)
  cat("--- MatchIt ---\n")

  for (n in c(100, 1000, 10000)) {
    df <- generate_treatment_data(n)
    local_df <- df

    # Nearest neighbor matching
    r <- run_bench(sprintf("matchit_nn_n%d", n),
                   function() matchit(treatment ~ x1 + x2 + x3, data = local_df, method = "nearest"),
                   iterations = 20)
    r$n <- n; r$method <- "matchit_nearest"; results[[idx]] <- r; idx <- idx + 1
    cat(sprintf("  nearest n=%d: %.2f us\n", n, r$time_median_us))

    # Full matching
    r <- run_bench(sprintf("matchit_full_n%d", n),
                   function() matchit(treatment ~ x1 + x2 + x3, data = local_df, method = "full"),
                   iterations = 20)
    r$n <- n; r$method <- "matchit_full"; results[[idx]] <- r; idx <- idx + 1
    cat(sprintf("  full n=%d: %.2f us\n", n, r$time_median_us))

    # Optimal matching (smaller sizes only)
    if (n <= 1000) {
      r <- run_bench(sprintf("matchit_opt_n%d", n),
                     function() matchit(treatment ~ x1 + x2 + x3, data = local_df, method = "optimal"),
                     iterations = 10)
      r$n <- n; r$method <- "matchit_optimal"; results[[idx]] <- r; idx <- idx + 1
      cat(sprintf("  optimal n=%d: %.2f us\n", n, r$time_median_us))
    }
  }
}

# WeightIt
if (requireNamespace("WeightIt", quietly = TRUE)) {
  library(WeightIt)
  cat("\n--- WeightIt ---\n")

  for (n in c(100, 1000, 10000)) {
    df <- generate_treatment_data(n)
    local_df <- df

    # Propensity score weighting
    r <- run_bench(sprintf("weightit_ps_n%d", n),
                   function() weightit(treatment ~ x1 + x2 + x3, data = local_df, method = "ps"),
                   iterations = 20)
    r$n <- n; r$method <- "weightit_ps"; results[[idx]] <- r; idx <- idx + 1
    cat(sprintf("  ps n=%d: %.2f us\n", n, r$time_median_us))

    # Entropy balancing
    r <- run_bench(sprintf("weightit_ebal_n%d", n),
                   function() weightit(treatment ~ x1 + x2 + x3, data = local_df, method = "ebal"),
                   iterations = 20)
    r$n <- n; r$method <- "weightit_ebal"; results[[idx]] <- r; idx <- idx + 1
    cat(sprintf("  ebal n=%d: %.2f us\n", n, r$time_median_us))

    # CBPS
    r <- run_bench(sprintf("weightit_cbps_n%d", n),
                   function() weightit(treatment ~ x1 + x2 + x3, data = local_df, method = "cbps"),
                   iterations = 20)
    r$n <- n; r$method <- "weightit_cbps"; results[[idx]] <- r; idx <- idx + 1
    cat(sprintf("  cbps n=%d: %.2f us\n", n, r$time_median_us))
  }
}

if (length(results) > 0) {
  results_df <- do.call(rbind, lapply(results, function(r) {
    data.frame(method = r$method, n = r$n, iterations = r$iterations,
      time_min_us = r$time_min_us, time_p25_us = r$time_p25_us,
      time_median_us = r$time_median_us, time_p75_us = r$time_p75_us,
      time_max_us = r$time_max_us, time_mean_us = r$time_mean_us,
      time_std_us = r$time_std_us, itr_per_sec = r$itr_per_sec,
      mem_alloc_bytes = r$mem_alloc_bytes)
  }))
  dir.create("results", showWarnings = FALSE)
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  write.csv(results_df, sprintf("results/r_matching_%s.csv", timestamp), row.names = FALSE)
}

cat("\nDone.\n")
