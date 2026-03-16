#!/usr/bin/env Rscript
# Mixed Logit (Random Parameters Logit) R Benchmark

# Install if not present
if (!require("gmnl", quietly = TRUE)) {
  install.packages("gmnl", repos = "https://cloud.r-project.org/")
}
if (!require("mlogit", quietly = TRUE)) {
  install.packages("mlogit", repos = "https://cloud.r-project.org/")
}

library(gmnl)
library(mlogit)

set.seed(42)

# Generate choice data with random preference heterogeneity
generate_mixedlogit_data <- function(n_choosers, n_alts) {
  n_obs <- n_choosers * n_alts

  chooser_ids <- rep(1:n_choosers, each = n_alts)
  alt_ids <- rep(1:n_alts, n_choosers)

  # Generate prices (alternative-specific)
  price <- runif(n_obs, 10, 30)
  time <- runif(n_obs, 5, 60)

  # Individual-specific random preference for price (heterogeneity)
  # True: beta_price ~ N(-0.1, 0.03)
  beta_price <- rep(rnorm(n_choosers, -0.1, 0.03), each = n_alts)
  beta_time <- -0.02  # Fixed coefficient for time

  # Generate choices based on utilities
  chosen <- numeric(n_obs)

  for (c in 1:n_choosers) {
    idx <- ((c - 1) * n_alts + 1):(c * n_alts)
    utilities <- beta_price[idx] * price[idx] + beta_time * time[idx] + rlogis(n_alts)
    chosen_alt <- which.max(utilities)
    chosen[idx] <- ifelse(1:n_alts == chosen_alt, 1, 0)
  }

  data.frame(
    chooser = chooser_ids,
    alt = alt_ids,
    price = price,
    time = time,
    chosen = chosen
  )
}

# Benchmark function for gmnl
benchmark_gmnl <- function(n_choosers, n_alts, n_draws = 100, n_reps = 3) {
  df <- generate_mixedlogit_data(n_choosers, n_alts)
  mdata <- mlogit.data(df, choice = "chosen", shape = "long",
                       chid.var = "chooser", alt.var = "alt")

  times <- numeric(n_reps)
  for (i in 1:n_reps) {
    start <- Sys.time()
    result <- gmnl(
      chosen ~ price + time | 0,
      data = mdata,
      model = "mixl",
      R = n_draws,
      ranp = c(price = "n"),  # price is normally distributed
      halton = NA,  # Use Halton sequences
      print.level = 0
    )
    end <- Sys.time()
    times[i] <- as.numeric(end - start, units = "secs")
  }

  list(
    median = median(times) * 1000,
    mean = mean(times) * 1000,
    sd = sd(times) * 1000
  )
}

cat("=== Mixed Logit (gmnl) R Benchmarks ===\n\n")

cat("Testing with different data sizes and draw counts:\n\n")

configs <- list(
  list(n = 100, alts = 3, draws = 100),
  list(n = 1000, alts = 3, draws = 100),
  list(n = 10000, alts = 3, draws = 100)
)

cat(sprintf("%-35s %15s %15s %15s\n", "Config", "Median (ms)", "Mean (ms)", "SD (ms)"))
cat(paste(rep("-", 85), collapse = ""), "\n")

for (cfg in configs) {
  result <- tryCatch({
    benchmark_gmnl(cfg$n, cfg$alts, cfg$draws, n_reps = 3)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    list(median = NA, mean = NA, sd = NA)
  })

  obs <- cfg$n * cfg$alts
  cat(sprintf("n=%d, alts=%d, draws=%d (%d obs) %10.2f %15.2f %15.2f\n",
              cfg$n, cfg$alts, cfg$draws, obs, result$median, result$mean, result$sd))
}

cat("\nNote: Mixed logit with one random parameter (price ~ Normal).\n")
cat("Halton sequences used for simulation.\n")
