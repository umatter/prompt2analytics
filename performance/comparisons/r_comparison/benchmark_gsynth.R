#!/usr/bin/env Rscript
# Generalized Synthetic Control (gsynth) R Benchmark
# Compares R implementation performance against p2a Rust

library(gsynth)

set.seed(42)

# Generate panel data for benchmarking
generate_panel_data <- function(n_units, n_times, n_treated_frac = 0.2) {
  n_treated <- max(1, floor(n_units * n_treated_frac))
  n_control <- n_units - n_treated

  units <- c()
  times <- c()
  outcomes <- c()
  treatment <- c()

  # Control units
  for (i in 1:n_control) {
    for (t in 1:n_times) {
      units <- c(units, paste0("C", i))
      times <- c(times, t)
      # Outcome with unit FE, time trend, and noise
      outcomes <- c(outcomes, 10 + i * 0.5 + t + rnorm(1, 0, 0.5))
      treatment <- c(treatment, 0)
    }
  }

  # Treated units (treatment starts at random times in second half)
  for (i in 1:n_treated) {
    treat_time <- floor(n_times / 2) + sample(1:(n_times / 2), 1)
    for (t in 1:n_times) {
      units <- c(units, paste0("T", i))
      times <- c(times, t)
      base <- 10 + (n_control + i) * 0.5 + t + rnorm(1, 0, 0.5)
      effect <- ifelse(t >= treat_time, 3 + runif(1, -1, 1), 0)
      outcomes <- c(outcomes, base + effect)
      treatment <- c(treatment, ifelse(t >= treat_time, 1, 0))
    }
  }

  data.frame(
    unit = units,
    time = times,
    outcome = outcomes,
    treated = treatment
  )
}

# Benchmark function
benchmark_gsynth <- function(n_units, n_times, n_reps = 10) {
  panel <- generate_panel_data(n_units, n_times)

  times <- numeric(n_reps)
  for (i in 1:n_reps) {
    start <- Sys.time()
    result <- gsynth(
      outcome ~ treated,
      data = panel,
      index = c("unit", "time"),
      force = "unit",
      r = 1,
      CV = FALSE,
      se = FALSE
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

cat("=== Generalized Synthetic Control (gsynth) R Benchmarks ===\n\n")

# Test configurations
configs <- list(
  list(n = 10, T = 10),    # Small (n=100)
  list(n = 50, T = 20),    # Medium (n=1000)
  list(n = 100, T = 100)   # Large (n=10000)
)

cat(sprintf("%-20s %15s %15s %15s\n", "Config", "Median (ms)", "Mean (ms)", "SD (ms)"))
cat(paste(rep("-", 70), collapse = ""), "\n")

for (cfg in configs) {
  result <- tryCatch({
    benchmark_gsynth(cfg$n, cfg$T, n_reps = 10)
  }, error = function(e) {
    list(median = NA, mean = NA, sd = NA)
  })

  cat(sprintf("n=%d, T=%d %15.2f %15.2f %15.2f\n",
              cfg$n, cfg$T, result$median, result$mean, result$sd))
}

cat("\nNote: Times are for single gsynth() calls with 1 factor, no CV, no SE.\n")
