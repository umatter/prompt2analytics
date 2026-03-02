#!/usr/bin/env Rscript
# generate_synthetic.R - Generate synthetic datasets for benchmarking
# Usage: Rscript generate_synthetic.R [output_dir]

suppressPackageStartupMessages({
  library(MASS)
})

# Parse arguments
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "."

# Set seed for reproducibility
set.seed(42)

# Sample sizes
sample_sizes <- c(100, 1000, 10000, 100000)

cat("Generating synthetic datasets...\n")

# Helper function to save dataset
save_dataset <- function(data, name, n) {
  filename <- file.path(output_dir, sprintf("synthetic_%s_n%d.csv", name, n))
  write.csv(data, filename, row.names = FALSE)
  cat(sprintf("  Saved: %s (%d rows)\n", filename, nrow(data)))
}

for (n in sample_sizes) {
  cat(sprintf("\n=== Sample size: %d ===\n", n))

  # 1. OLS dataset: y = 2 + 3*x1 - 1.5*x2 + 0.5*x3 + epsilon
  x1 <- rnorm(n, mean = 5, sd = 2)
  x2 <- rnorm(n, mean = 10, sd = 3)
  x3 <- rnorm(n, mean = 0, sd = 1)
  epsilon <- rnorm(n, mean = 0, sd = 2)
  y <- 2 + 3*x1 - 1.5*x2 + 0.5*x3 + epsilon

  ols_data <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
  save_dataset(ols_data, "ols", n)

  # 2. Panel dataset with entity and time effects
  n_entities <- max(10, n / 20)
  n_time <- 20
  entity_id <- rep(1:n_entities, each = n_time)
  time_id <- rep(1:n_time, times = n_entities)
  n_panel <- n_entities * n_time

  entity_effect <- rep(rnorm(n_entities, sd = 2), each = n_time)
  time_effect <- rep(rnorm(n_time, sd = 1), times = n_entities)

  x1_panel <- rnorm(n_panel, mean = 5, sd = 2)
  x2_panel <- rnorm(n_panel, mean = 10, sd = 3)
  epsilon_panel <- rnorm(n_panel, mean = 0, sd = 1)
  y_panel <- 1 + 2*x1_panel - 0.5*x2_panel + entity_effect + time_effect + epsilon_panel

  panel_data <- data.frame(
    entity = entity_id, time = time_id,
    y = y_panel, x1 = x1_panel, x2 = x2_panel
  )
  save_dataset(panel_data, "panel", n_panel)

  # 3. Clustered data
  n_clusters <- max(20, n / 50)
  cluster_size <- n / n_clusters
  cluster_id <- rep(1:n_clusters, each = cluster_size)
  cluster_effect <- rep(rnorm(n_clusters, sd = 3), each = cluster_size)

  x1_clust <- rnorm(n, mean = 5, sd = 2)
  x2_clust <- rnorm(n, mean = 10, sd = 3)
  epsilon_clust <- rnorm(n, mean = 0, sd = 1)
  y_clust <- 2 + 3*x1_clust - 1.5*x2_clust + cluster_effect + epsilon_clust

  clust_data <- data.frame(
    cluster = cluster_id,
    y = y_clust, x1 = x1_clust, x2 = x2_clust
  )
  save_dataset(clust_data, "clustered", n)

  # 4. Binary outcome for logit/probit
  x1_bin <- rnorm(n, mean = 0, sd = 1)
  x2_bin <- rnorm(n, mean = 0, sd = 1)
  latent <- -1 + 2*x1_bin - 1*x2_bin + rnorm(n, sd = 1)
  y_bin <- as.integer(latent > 0)

  binary_data <- data.frame(y = y_bin, x1 = x1_bin, x2 = x2_bin)
  save_dataset(binary_data, "binary", n)

  # 5. IV dataset with endogenous variable
  z1 <- rnorm(n, mean = 0, sd = 1)  # instrument
  z2 <- rnorm(n, mean = 0, sd = 1)  # instrument
  u <- rnorm(n, mean = 0, sd = 1)   # unobserved

  x_endog <- 1 + 0.5*z1 + 0.3*z2 + 0.7*u + rnorm(n, sd = 0.5)  # endogenous
  x_exog <- rnorm(n, mean = 5, sd = 2)  # exogenous
  y_iv <- 2 + 1.5*x_endog + 0.8*x_exog + 0.5*u + rnorm(n, sd = 1)

  iv_data <- data.frame(
    y = y_iv, x_endog = x_endog, x_exog = x_exog,
    z1 = z1, z2 = z2
  )
  save_dataset(iv_data, "iv", n)

  # 6. Time series (ARIMA-like)
  ts_data <- arima.sim(n = n, model = list(ar = 0.7, ma = 0.3))
  ts_df <- data.frame(y = as.numeric(ts_data), t = 1:n)
  save_dataset(ts_df, "timeseries", n)

  # 7. Clustering data (multivariate)
  k_true <- 3
  centers <- matrix(c(0, 0, 5, 5, 10, 0), nrow = 3, byrow = TRUE)
  # Ensure cluster sizes sum exactly to n
  base_size <- floor(n / k_true)
  remainder <- n %% k_true
  cluster_sizes <- rep(base_size, k_true)
  if (remainder > 0) {
    cluster_sizes[1:remainder] <- cluster_sizes[1:remainder] + 1
  }
  cluster_labels <- rep(1:k_true, cluster_sizes)

  X_cluster <- matrix(0, nrow = n, ncol = 2)
  for (i in 1:k_true) {
    idx <- which(cluster_labels == i)
    X_cluster[idx, ] <- mvrnorm(length(idx), mu = centers[i, ], Sigma = diag(c(1, 1)))
  }

  cluster_data <- data.frame(x1 = X_cluster[, 1], x2 = X_cluster[, 2], true_cluster = cluster_labels)
  save_dataset(cluster_data, "cluster", n)

  # 8. PCA data (correlated variables)
  Sigma <- matrix(c(1.0, 0.8, 0.6, 0.4,
                    0.8, 1.0, 0.7, 0.5,
                    0.6, 0.7, 1.0, 0.6,
                    0.4, 0.5, 0.6, 1.0), nrow = 4)
  X_pca <- mvrnorm(n, mu = c(0, 0, 0, 0), Sigma = Sigma)
  pca_data <- data.frame(x1 = X_pca[, 1], x2 = X_pca[, 2], x3 = X_pca[, 3], x4 = X_pca[, 4])
  save_dataset(pca_data, "pca", n)

  # 9. Survival data (time-to-event with censoring)
  x1_surv <- rnorm(n, mean = 0, sd = 1)
  x2_surv <- rnorm(n, mean = 0, sd = 1)
  # Exponential baseline hazard with covariates
  rate <- exp(-2 + 0.5 * x1_surv - 0.3 * x2_surv)
  time_surv <- rexp(n, rate = rate)
  # Right censoring at random times
  censor_time <- rexp(n, rate = 0.1)
  observed_time <- pmin(time_surv, censor_time)
  event <- as.integer(time_surv <= censor_time)

  surv_data <- data.frame(
    time = observed_time, event = event,
    x1 = x1_surv, x2 = x2_surv
  )
  save_dataset(surv_data, "survival", n)

  # 10. Count data (Poisson/NegBin)
  x1_count <- rnorm(n, mean = 0, sd = 1)
  x2_count <- rnorm(n, mean = 0, sd = 1)
  lambda <- exp(1 + 0.5 * x1_count - 0.3 * x2_count)
  # Negative binomial with overdispersion
  y_count <- rnbinom(n, size = 2, mu = lambda)

  count_data <- data.frame(y = y_count, x1 = x1_count, x2 = x2_count)
  save_dataset(count_data, "count", n)

  # 11. Factor/ANOVA data (groups for hypothesis tests)
  n_groups <- 4
  group_size <- n / n_groups
  group <- rep(paste0("G", 1:n_groups), each = group_size)
  group_means <- c(5, 7, 6, 8)
  y_factor <- unlist(lapply(1:n_groups, function(i) {
    rnorm(group_size, mean = group_means[i], sd = 2)
  }))
  # Second factor for two-way ANOVA
  factor2 <- rep(c("A", "B"), times = n / 2)

  factor_data <- data.frame(
    y = y_factor, group = group, factor2 = factor2,
    x1 = rnorm(n)
  )
  save_dataset(factor_data, "factor", n)

  # 12. Multivariate time series (for VAR/Granger)
  y1 <- numeric(n)
  y2 <- numeric(n)
  y1[1] <- rnorm(1)
  y2[1] <- rnorm(1)
  for (i in 2:n) {
    y1[i] <- 0.7 * y1[i-1] + 0.2 * y2[i-1] + rnorm(1, sd = 0.5)
    y2[i] <- 0.3 * y1[i-1] + 0.6 * y2[i-1] + rnorm(1, sd = 0.5)
  }

  mvar_data <- data.frame(y1 = y1, y2 = y2, t = 1:n)
  save_dataset(mvar_data, "multivar_ts", n)
}

cat("\nDone!\n")
