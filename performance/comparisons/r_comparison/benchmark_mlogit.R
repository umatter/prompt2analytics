#!/usr/bin/env Rscript
# McFadden Conditional Logit (mlogit) R Benchmark

library(mlogit)

set.seed(42)

# Generate choice data
generate_mlogit_data <- function(n_choosers, n_alts) {
  n_obs <- n_choosers * n_alts

  chooser_ids <- rep(1:n_choosers, each = n_alts)
  alt_ids <- rep(1:n_alts, n_choosers)

  # Generate prices (alternative-specific)
  price <- runif(n_obs, 10, 30)

  # True utility: U = -0.1 * price + error
  # Generate choices based on logit probabilities
  chosen <- numeric(n_obs)

  for (c in 1:n_choosers) {
    idx <- ((c - 1) * n_alts + 1):(c * n_alts)
    utilities <- -0.1 * price[idx] + rlogis(n_alts)
    chosen_alt <- which.max(utilities)
    chosen[idx] <- ifelse(1:n_alts == chosen_alt, 1, 0)
  }

  data.frame(
    chooser = chooser_ids,
    alt = alt_ids,
    price = price,
    chosen = chosen
  )
}

# Benchmark function
benchmark_mlogit <- function(n_choosers, n_alts, n_reps = 10) {
  df <- generate_mlogit_data(n_choosers, n_alts)
  mdata <- dfidx(df, choice = "chosen", idx = c("chooser", "alt"))

  times <- numeric(n_reps)
  for (i in 1:n_reps) {
    start <- Sys.time()
    result <- mlogit(chosen ~ price | 0, data = mdata)
    end <- Sys.time()
    times[i] <- as.numeric(end - start, units = "secs")
  }

  list(
    median = median(times) * 1000,  # Convert to ms
    mean = mean(times) * 1000,
    sd = sd(times) * 1000
  )
}

cat("=== McFadden Conditional Logit (mlogit) R Benchmarks ===\n\n")

configs <- list(
  list(n = 100, alts = 3),
  list(n = 1000, alts = 3),
  list(n = 10000, alts = 3)
)

cat(sprintf("%-25s %15s %15s %15s\n", "Config", "Median (ms)", "Mean (ms)", "SD (ms)"))
cat(paste(rep("-", 75), collapse = ""), "\n")

for (cfg in configs) {
  result <- tryCatch({
    benchmark_mlogit(cfg$n, cfg$alts, n_reps = 10)
  }, error = function(e) {
    list(median = NA, mean = NA, sd = NA)
  })

  obs <- cfg$n * cfg$alts
  cat(sprintf("n=%d, alts=%d (%d obs) %10.2f %15.2f %15.2f\n",
              cfg$n, cfg$alts, obs, result$median, result$mean, result$sd))
}

cat("\nNote: Times are for mlogit() with alternative-specific price variable.\n")
