#!/usr/bin/env Rscript
# Extended Two-Way Fixed Effects (ETWFE) R Benchmark

# Install if not present
if (!require("etwfe", quietly = TRUE)) {
  install.packages("etwfe", repos = "https://cloud.r-project.org/")
}
if (!require("fixest", quietly = TRUE)) {
  install.packages("fixest", repos = "https://cloud.r-project.org/")
}

library(etwfe)
library(fixest)

set.seed(42)

# Generate staggered DiD panel data
generate_etwfe_data <- function(n_units, n_periods, treat_share = 0.5) {
  # Create panel structure
  unit <- rep(1:n_units, each = n_periods)
  time <- rep(1:n_periods, times = n_units)

  # Assign treatment timing (staggered adoption)
  n_treated <- floor(n_units * treat_share)
  treatment_times <- sample(3:(n_periods-1), n_treated, replace = TRUE)
  first_treat <- c(treatment_times, rep(0, n_units - n_treated))
  first_treat <- first_treat[sample(n_units)]  # Shuffle
  first_treat_expanded <- rep(first_treat, each = n_periods)

  # Create treatment indicator
  treat <- as.integer(time >= first_treat_expanded & first_treat_expanded > 0)

  # Generate outcome with treatment effect
  # True ATT varies by cohort: earlier cohorts have larger effect
  cohort_effect <- ifelse(first_treat_expanded > 0, 10 - 0.5 * first_treat_expanded, 0)

  # Fixed effects + treatment + noise
  unit_fe <- rep(rnorm(n_units, 0, 2), each = n_periods)
  time_fe <- rep(rnorm(n_periods, 0, 1), times = n_units)
  noise <- rnorm(n_units * n_periods, 0, 1)

  y <- unit_fe + time_fe + treat * cohort_effect + noise

  data.frame(
    unit = unit,
    time = time,
    y = y,
    treat = treat,
    first_treat = first_treat_expanded
  )
}

# Benchmark function
benchmark_etwfe <- function(n_units, n_periods, n_reps = 5) {
  df <- generate_etwfe_data(n_units, n_periods)

  times <- numeric(n_reps)
  for (i in 1:n_reps) {
    start <- Sys.time()
    result <- etwfe(
      fml = y ~ 1,
      tvar = time,
      gvar = first_treat,
      data = df,
      vcov = "HC1"
    )
    end <- Sys.time()
    times[i] <- as.numeric(end - start, units = "secs")
  }

  list(
    median = median(times) * 1000,  # Convert to ms
    mean = mean(times) * 1000,
    sd = sd(times) * 1000
  )
}

cat("=== Extended Two-Way Fixed Effects (ETWFE) R Benchmarks ===\n\n")

configs <- list(
  list(units = 50, periods = 10),
  list(units = 100, periods = 10),
  list(units = 100, periods = 20),
  list(units = 200, periods = 15),
  list(units = 500, periods = 10),
  list(units = 500, periods = 20)
)

cat(sprintf("%-30s %15s %15s %15s\n", "Config", "Median (ms)", "Mean (ms)", "SD (ms)"))
cat(paste(rep("-", 80), collapse = ""), "\n")

for (cfg in configs) {
  result <- tryCatch({
    benchmark_etwfe(cfg$units, cfg$periods, n_reps = 5)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    list(median = NA, mean = NA, sd = NA)
  })

  obs <- cfg$units * cfg$periods
  cat(sprintf("units=%d, periods=%d (%d obs) %10.2f %15.2f %15.2f\n",
              cfg$units, cfg$periods, obs, result$median, result$mean, result$sd))
}

cat("\nNote: Times include ETWFE estimation with HC1 standard errors.\n")
