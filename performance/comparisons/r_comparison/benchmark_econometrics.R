#!/usr/bin/env Rscript
# Econometrics Benchmarks for Cross-Language Comparison
# Compares R packages (plm, lfe, AER) against p2a Rust implementation

library(microbenchmark)
library(plm)
library(lfe)

# Ensure reproducibility
set.seed(42)

# Generate panel data with same DGP as Rust benchmarks
generate_panel_data <- function(n_entities, n_periods) {
  n <- n_entities * n_periods

  entity <- rep(1:n_entities, each = n_periods)
  time <- rep(1:n_periods, times = n_entities)

  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)

  # y = 1.0*x1 + 0.5*x2 + entity_effect + noise
  entity_effect <- entity * 0.5
  y <- 1.0 * x1 + 0.5 * x2 + entity_effect + runif(n, 0, 0.3)

  data.frame(y = y, x1 = x1, x2 = x2, entity = factor(entity), time = factor(time))
}

# Generate binary outcome data
generate_binary_data <- function(n) {
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)

  # Probability via logit
  latent <- 0.5 * x1 + 0.3 * x2
  prob <- 1 / (1 + exp(-latent))
  y <- rbinom(n, 1, prob)

  data.frame(y = y, x1 = x1, x2 = x2)
}

# Benchmark Fixed Effects
benchmark_fixed_effects <- function() {
  results <- list()

  configs <- list(
    c(10, 10),
    c(50, 20),
    c(100, 50)
  )

  for (cfg in configs) {
    n_entities <- cfg[1]
    n_periods <- cfg[2]
    n <- n_entities * n_periods

    cat(sprintf("Benchmarking FE with n=%d (entities=%d, periods=%d)\n",
                n, n_entities, n_periods))

    data <- generate_panel_data(n_entities, n_periods)
    pdata <- pdata.frame(data, index = c("entity", "time"))

    # plm Fixed Effects
    bm_plm <- microbenchmark(
      plm(y ~ x1 + x2, data = pdata, model = "within"),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("fe_plm_", n)]] <- summary(bm_plm)

    # lfe Fixed Effects (felm)
    bm_lfe <- microbenchmark(
      felm(y ~ x1 + x2 | entity, data = data),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("fe_lfe_", n)]] <- summary(bm_lfe)
  }

  results
}

# Benchmark HDFE (two-way fixed effects)
benchmark_hdfe <- function() {
  results <- list()

  configs <- list(
    c(10, 10),
    c(50, 20),
    c(100, 50)
  )

  for (cfg in configs) {
    n_entities <- cfg[1]
    n_periods <- cfg[2]
    n <- n_entities * n_periods

    cat(sprintf("Benchmarking HDFE with n=%d\n", n))

    data <- generate_panel_data(n_entities, n_periods)

    bm <- microbenchmark(
      felm(y ~ x1 + x2 | entity + time, data = data),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("hdfe_", n)]] <- summary(bm)
  }

  results
}

# Benchmark Logit
benchmark_logit <- function() {
  results <- list()

  for (n in c(100, 500, 1000)) {
    cat(sprintf("Benchmarking Logit with n=%d\n", n))
    data <- generate_binary_data(n)

    bm <- microbenchmark(
      glm(y ~ x1 + x2, data = data, family = binomial(link = "logit")),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("logit_", n)]] <- summary(bm)
  }

  results
}

# Benchmark Probit
benchmark_probit <- function() {
  results <- list()

  for (n in c(100, 500, 1000)) {
    cat(sprintf("Benchmarking Probit with n=%d\n", n))
    data <- generate_binary_data(n)

    bm <- microbenchmark(
      glm(y ~ x1 + x2, data = data, family = binomial(link = "probit")),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("probit_", n)]] <- summary(bm)
  }

  results
}

# Run benchmarks
cat("=== Fixed Effects Benchmarks ===\n")
fe_results <- benchmark_fixed_effects()

cat("\n=== HDFE Benchmarks ===\n")
hdfe_results <- benchmark_hdfe()

cat("\n=== Logit Benchmarks ===\n")
logit_results <- benchmark_logit()

cat("\n=== Probit Benchmarks ===\n")
probit_results <- benchmark_probit()

# Save results
save_results <- function(results, filename) {
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

save_results(fe_results, "results/econometrics_fe.csv")
save_results(hdfe_results, "results/econometrics_hdfe.csv")
save_results(logit_results, "results/econometrics_logit.csv")
save_results(probit_results, "results/econometrics_probit.csv")

# Print summary
cat("\n=== Summary ===\n")
cat("Fixed Effects:\n")
print(do.call(rbind, fe_results))
cat("\nHDFE:\n")
print(do.call(rbind, hdfe_results))
cat("\nLogit:\n")
print(do.call(rbind, logit_results))
cat("\nProbit:\n")
print(do.call(rbind, probit_results))
