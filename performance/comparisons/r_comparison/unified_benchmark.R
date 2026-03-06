#!/usr/bin/env Rscript
# Unified R Benchmark: Data generation + timing + output capture
# Generates data CSVs for Rust to load, then benchmarks R methods
#
# Usage: Rscript unified_benchmark.R [--data-dir ../data]
#
# Install required packages:
# install.packages(c("bench", "jsonlite", "sandwich", "plm", "lfe", "forecast",
#                     "changepoint", "survival", "MatchIt", "WeightIt", "CBPS",
#                     "tmle", "ctmle", "mediation", "sensemakr", "bacondecomp", "did",
#                     "spdep", "spatialreg", "randomForest", "dbscan",
#                     "DoubleML", "mlr3", "mlr3learners", "rdrobust",
#                     "gsynth", "cluster", "e1071", "Rtsne", "cmprsk",
#                     "fixest", "EValue", "margins", "fGarch"))

# ============================================
# Parse arguments
# ============================================
args <- commandArgs(trailingOnly = TRUE)
data_dir <- "../data"
for (i in seq_along(args)) {
  if (args[i] == "--data-dir" && i < length(args)) data_dir <- args[i + 1]
}

# ============================================
# Load packages
# ============================================
suppressPackageStartupMessages({
  library(bench)
  library(jsonlite)
  library(sandwich)
  library(lmtest)
})

dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

results <- list()

# ============================================
# Extractor functions
# ============================================
extract_lm <- function(fit) {
  s <- summary(fit)
  list(
    coefficients = as.numeric(coef(fit)),
    std_errors = as.numeric(s$coefficients[, "Std. Error"]),
    r_squared = s$r.squared,
    f_statistic = as.numeric(s$fstatistic[1])
  )
}

extract_lm_hc1 <- function(result) {
  # result is a coeftest object
  list(
    coefficients = as.numeric(result[, "Estimate"]),
    std_errors = as.numeric(result[, "Std. Error"])
  )
}

extract_glm <- function(fit) {
  s <- summary(fit)
  list(
    coefficients = as.numeric(coef(fit)),
    std_errors = as.numeric(s$coefficients[, "Std. Error"]),
    log_likelihood = as.numeric(logLik(fit))
  )
}

extract_plm <- function(fit) {
  s <- summary(fit)
  list(
    coefficients = as.numeric(coef(fit)),
    std_errors = as.numeric(s$coefficients[, "Std. Error"]),
    r_squared = s$r.squared[1]
  )
}

extract_felm <- function(fit) {
  s <- summary(fit)
  list(
    coefficients = as.numeric(coef(fit)),
    std_errors = as.numeric(s$coefficients[, "Std. Error"]),
    r_squared = s$r2
  )
}

extract_test <- function(result) {
  list(
    statistic = as.numeric(result$statistic),
    p_value = as.numeric(result$p.value)
  )
}

extract_kmeans <- function(result) {
  list(
    within_ss = result$tot.withinss,
    n_clusters = length(result$size),
    cluster_sizes = as.numeric(sort(result$size))
  )
}

extract_pca <- function(result) {
  list(
    eigenvalues = as.numeric(result$sdev^2),
    variance_explained = as.numeric(summary(result)$importance[2, ])
  )
}

extract_arima <- function(fit) {
  cc <- coef(fit)
  ar_names <- grep("^ar", names(cc), value = TRUE)
  ma_names <- grep("^ma", names(cc), value = TRUE)
  intercept_names <- grep("^intercept$|^drift$|^mean$", names(cc), value = TRUE)
  list(
    ar_coef = as.numeric(cc[ar_names]),
    ma_coef = as.numeric(cc[ma_names]),
    intercept = if (length(intercept_names) > 0) as.numeric(cc[intercept_names[1]]) else 0.0,
    aic = fit$aic
  )
}

extract_survfit <- function(fit) {
  list(
    n_events = sum(fit$n.event),
    median_survival = as.numeric(summary(fit)$table["median"])
  )
}

extract_coxph <- function(fit) {
  s <- summary(fit)
  list(
    coefficients = as.numeric(coef(fit)),
    std_errors = as.numeric(s$coefficients[, "se(coef)"]),
    log_likelihood = as.numeric(fit$loglik[2])
  )
}

# ============================================
# Main benchmark runner
# ============================================
run_unified <- function(method, variant, n, seed, dgp, fn, extract_fn, iterations = 50) {
  message(sprintf("  Benchmarking %s/%s n=%d ...", method, variant, n))

  bm <- tryCatch(
    bench::mark(fn(), iterations = iterations, check = FALSE, memory = TRUE, filter_gc = FALSE),
    error = function(e) {
      message(sprintf("    ERROR: %s", conditionMessage(e)))
      return(NULL)
    }
  )

  if (is.null(bm)) return(NULL)

  raw_times <- as.numeric(bm$time[[1]]) * 1e6  # seconds to microseconds

  # Capture outputs
  output <- tryCatch(fn(), error = function(e) NULL)
  outputs <- if (!is.null(output) && !is.null(extract_fn)) {
    tryCatch(extract_fn(output), error = function(e) list())
  } else {
    list()
  }

  list(
    method = method,
    variant = variant,
    n = n,
    seed = seed,
    dgp = dgp,
    language = "R",
    iterations = length(raw_times),
    time_min_us = min(raw_times),
    time_p25_us = as.numeric(quantile(raw_times, 0.25)),
    time_p75_us = as.numeric(quantile(raw_times, 0.75)),
    time_median_us = median(raw_times),
    time_max_us = max(raw_times),
    time_mean_us = mean(raw_times),
    time_std_us = sd(raw_times),
    itr_per_sec = 1e6 / median(raw_times),
    mem_alloc_bytes = as.numeric(bm$mem_alloc[[1]]),
    outputs = outputs
  )
}

# Helper to save CSV and return data
save_dgp <- function(df, dgp_name) {
  path <- file.path(data_dir, paste0(dgp_name, ".csv"))
  write.csv(df, file = path, row.names = FALSE)
  message(sprintf("  Saved: %s", path))
  df
}

# ============================================
# Data Generators (matching Rust DGPs with seed=42)
# ============================================

generate_regression_data <- function(n, k = 5) {
  set.seed(42)
  X <- matrix(runif(n * k, -1, 1), nrow = n, ncol = k)
  colnames(X) <- paste0("x", 1:k)
  y <- rowSums(X) + runif(n, 0, 0.5)
  data.frame(y = y, X)
}

generate_panel_data <- function(n_entities, n_periods) {
  set.seed(42)
  n <- n_entities * n_periods
  entity <- rep(0:(n_entities - 1), each = n_periods)
  time <- rep(0:(n_periods - 1), times = n_entities)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  entity_effect <- entity * 0.1
  y <- entity_effect + 0.5 * x1 + 0.3 * x2 + runif(n, 0, 0.5)
  data.frame(entity = entity, time = time, y = y, x1 = x1, x2 = x2)
}

generate_binary_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  linear <- -1 + 0.5 * x1 + 0.3 * x2
  prob <- 1 / (1 + exp(-linear))
  y <- as.numeric(runif(n) < prob)
  data.frame(y = y, x1 = x1, x2 = x2)
}

generate_time_series <- function(n) {
  set.seed(42)
  t <- 0:(n - 1)
  trend <- 0.01 * t
  seasonal <- sin(t * pi / 6) * 2.0
  noise <- runif(n, 0, 0.5)
  data.frame(y = trend + seasonal + noise)
}

generate_cluster_data <- function(n, k = 5) {
  set.seed(42)
  mat <- matrix(0, nrow = n, ncol = k)
  for (i in 1:n) {
    cluster <- (i - 1) %% 3
    center <- cluster * 3
    mat[i, ] <- center + runif(k, -0.5, 0.5)
  }
  colnames(mat) <- paste0("v", 1:k)
  as.data.frame(mat)
}

generate_did_data <- function(n) {
  set.seed(42)
  half <- n %/% 2
  treatment <- c(rep(0, half), rep(1, n - half))
  post <- rep(c(0, 1), length.out = n)
  x1 <- runif(n, -1, 1)
  y <- 1.0 + 0.5 * treatment + 0.3 * post + 2.0 * treatment * post + 0.4 * x1 + runif(n, -0.5, 0.5)
  data.frame(y = y, treatment = treatment, post = post, x1 = x1)
}

generate_iv_data <- function(n) {
  set.seed(42)
  z <- runif(n, -2, 2)
  x_exog <- runif(n, -1, 1)
  u <- runif(n, -1, 1)
  x_endog <- 0.5 * z + 0.3 * u + runif(n, -0.3, 0.3)
  y <- 1.0 + 0.8 * x_endog + 0.5 * x_exog + u + runif(n, -0.3, 0.3)
  data.frame(y = y, x_exog = x_exog, x_endog = x_endog, instrument = z)
}

generate_rd_data <- function(n) {
  set.seed(42)
  running <- runif(n, -1, 1)
  te <- ifelse(running >= 0, 1.5, 0)
  y <- 0.5 + 0.3 * running + te + runif(n, -0.5, 0.5)
  data.frame(y = y, running = running)
}

generate_treatment_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  prob <- 1 / (1 + exp(-(-0.3 * x1 - 0.2 * x2)))
  treatment <- as.numeric(runif(n) < prob)
  y <- 1.0 + 0.5 * treatment + 0.3 * x1 + 0.2 * x2 + runif(n, -0.5, 0.5)
  data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2)
}

generate_staggered_panel <- function(n_units, n_periods) {
  set.seed(42)
  n <- n_units * n_periods
  unit <- integer(n)
  time <- integer(n)
  y <- numeric(n)
  treat_time <- integer(n)
  treated <- numeric(n)

  idx <- 1
  for (u in 0:(n_units - 1)) {
    tt <- if (u < n_units %/% 3) {
      0L  # never treated
    } else {
      as.integer((n_periods %/% 3) + (u %% (n_periods %/% 2)) + 1)
    }
    unit_effect <- u * 0.1
    for (t in 0:(n_periods - 1)) {
      unit[idx] <- u
      time[idx] <- t
      treat_time[idx] <- tt
      is_treated <- tt > 0 && t >= tt
      treated[idx] <- if (is_treated) 1.0 else 0.0
      te <- if (is_treated) 2.0 else 0.0
      y[idx] <- unit_effect + 0.05 * t + te + runif(1, -0.5, 0.5)
      idx <- idx + 1
    }
  }

  data.frame(unit = unit, time = time, y = y, treat_time = treat_time, treated = treated)
}

generate_spatial_data <- function(n_side) {
  set.seed(42)
  n <- n_side * n_side
  coords <- expand.grid(x_coord = 0:(n_side - 1), y_coord = 0:(n_side - 1))
  x <- runif(n, -1, 1)
  y_vals <- 2.0 + 0.7 * x + 0.3 * (coords$x_coord + coords$y_coord) / n_side + runif(n, -0.25, 0.25)
  data.frame(y = y_vals, x = x, x_coord = coords$x_coord, y_coord = coords$y_coord)
}

generate_survival_data <- function(n, censoring_rate = 0.3) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  group <- rep(c("A", "B"), length.out = n)
  linear <- 0.5 * x1 + 0.3 * x2 + ifelse(group == "B", 0.5, 0)
  u <- runif(n, 0.0001, 0.9999)
  shape <- 1.5
  scale_param <- 10.0
  true_time <- scale_param * (-log(u))^(1 / shape) * exp(-linear)
  censor_rate_adj <- censoring_rate * 0.1 * scale_param
  censor_time <- -censor_rate_adj * log(runif(n, 0.0001, 0.9999))
  time <- pmin(true_time, censor_time)
  event <- as.integer(true_time < censor_time)
  data.frame(time = time, event = event, x1 = x1, x2 = x2, group = group)
}

generate_mediation_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  treatment <- as.numeric(runif(n) < 0.5)
  mediator <- 0.5 * treatment + 0.3 * x1 + runif(n, -0.5, 0.5)
  y <- 1.0 + 0.3 * treatment + 0.5 * mediator + 0.2 * x1 + runif(n, -0.5, 0.5)
  data.frame(y = y, treatment = treatment, mediator = mediator, x1 = x1)
}

generate_factor_data <- function(n, p = 10) {
  set.seed(42)
  mat <- matrix(0, nrow = n, ncol = p)
  for (i in 1:n) {
    f1 <- runif(1, -2, 2)
    f2 <- runif(1, -2, 2)
    f3 <- runif(1, -2, 2)
    mat[i, 1] <- 0.8 * f1 + runif(1, -0.3, 0.3)
    mat[i, 2] <- 0.7 * f1 + runif(1, -0.4, 0.4)
    mat[i, 3] <- 0.75 * f1 + runif(1, -0.35, 0.35)
    mat[i, 4] <- 0.8 * f2 + runif(1, -0.3, 0.3)
    mat[i, 5] <- 0.7 * f2 + runif(1, -0.4, 0.4)
    mat[i, 6] <- 0.75 * f2 + runif(1, -0.35, 0.35)
    mat[i, 7] <- 0.8 * f3 + runif(1, -0.3, 0.3)
    mat[i, 8] <- 0.7 * f3 + runif(1, -0.4, 0.4)
    mat[i, 9] <- 0.75 * f3 + runif(1, -0.35, 0.35)
    mat[i, 10] <- runif(1, -1, 1)
  }
  colnames(mat) <- paste0("v", 1:p)
  as.data.frame(mat)
}

generate_count_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  group <- as.integer((seq_len(n) - 1) %% 5)
  lambda <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
  y <- rpois(n, lambda)
  data.frame(y = y, x1 = x1, x2 = x2, group = group)
}

generate_zeroinfl_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  lambda <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
  # 30% structural zeros
  zero_mask <- runif(n) < 0.3
  y <- rpois(n, lambda)
  y[zero_mask] <- 0L
  data.frame(y = y, x1 = x1, x2 = x2)
}

generate_ordered_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  u <- runif(n, 0.001, 0.999)
  logistic_noise <- log(u / (1 - u))
  y_star <- 0.5 * x1 + 0.3 * x2 + logistic_noise
  y <- ifelse(y_star < -0.5, "1", ifelse(y_star < 0.5, "2", "3"))
  data.frame(y = y, x1 = x1, x2 = x2, stringsAsFactors = FALSE)
}

generate_multinomial_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  u1 <- 0.5 * x1 + 0.2 * x2 + runif(n, -1, 1)
  u2 <- -0.3 * x1 + 0.4 * x2 + runif(n, -1, 1)
  u3 <- runif(n, -1, 1)
  y <- ifelse(u1 >= u2 & u1 >= u3, "A", ifelse(u2 >= u3, "B", "C"))
  data.frame(y = y, x1 = x1, x2 = x2, stringsAsFactors = FALSE)
}

generate_garch_data <- function(n) {
  set.seed(42)
  omega <- 0.1; alpha <- 0.15; beta <- 0.75
  y <- numeric(n); sigma2 <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)
  y[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * y[t-1]^2 + beta * sigma2[t-1]
    y[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  data.frame(y = y)
}

generate_bivariate_data <- function(n) {
  set.seed(42)
  y1 <- numeric(n); y2 <- numeric(n)
  y1[1] <- rnorm(1); y2[1] <- rnorm(1)
  for (t in 2:n) {
    y1[t] <- 0.5*y1[t-1] + 0.1*y2[t-1] + rnorm(1, 0, 0.5)
    y2[t] <- 0.2*y1[t-1] + 0.6*y2[t-1] + rnorm(1, 0, 0.5)
  }
  data.frame(y1=y1, y2=y2)
}

# ============================================
# PRE-GENERATE ALL DATA CSVs
# ============================================
# Generate data for ALL DGPs unconditionally so Rust can always load them,
# even when R analysis packages are not installed.
message("\n=== Pre-generating data CSVs ===")

# Regression
for (n in c(100, 1000, 10000)) save_dgp(generate_regression_data(n), sprintf("regression_n%d", n))

# Panel
for (params in list(c(10, 10), c(50, 20), c(100, 50))) {
  save_dgp(generate_panel_data(params[1], params[2]), sprintf("panel_n%d", params[1] * params[2]))
}

# Binary (all sizes Rust needs)
for (n in c(100, 200, 500, 1000)) save_dgp(generate_binary_data(n), sprintf("binary_n%d", n))

# Time series (all sizes Rust needs)
for (n in c(100, 200, 500, 1000)) save_dgp(generate_time_series(n), sprintf("timeseries_n%d", n))

# Cluster (all sizes Rust needs)
for (n in c(100, 500, 1000, 5000)) save_dgp(generate_cluster_data(n), sprintf("cluster_n%d", n))

# LOESS
for (n in c(100, 500, 1000)) save_dgp(generate_regression_data(n, 1), sprintf("loess_n%d", n))

# Factor analysis
for (n in c(100, 500, 1000)) save_dgp(generate_factor_data(n), sprintf("factor_n%d", n))

# DiD
for (n in c(200, 500, 1000)) save_dgp(generate_did_data(n), sprintf("did_n%d", n))

# IV
for (n in c(200, 500, 1000)) save_dgp(generate_iv_data(n), sprintf("iv_n%d", n))

# RD
for (n in c(200, 500, 1000)) save_dgp(generate_rd_data(n), sprintf("rd_n%d", n))

# Treatment (for IPW, TMLE, CTMLE, CBPS, Matching, WeightIt)
for (n in c(200, 500, 1000)) save_dgp(generate_treatment_data(n), sprintf("treatment_n%d", n))

# Staggered panel (for Staggered DiD, ETWFE, Bacon)
for (params in list(c(20, 10), c(50, 10))) {
  save_dgp(generate_staggered_panel(params[1], params[2]), sprintf("staggered_n%d", params[1] * params[2]))
}

# Mediation
for (n in c(200, 500, 1000)) save_dgp(generate_mediation_data(n), sprintf("mediation_n%d", n))

# Spatial
for (n_side in c(10, 20, 32)) save_dgp(generate_spatial_data(n_side), sprintf("spatial_n%d", n_side * n_side))

# Survival
for (n in c(100, 500, 1000)) save_dgp(generate_survival_data(n), sprintf("survival_n%d", n))

# DoubleML
for (n in c(200, 500, 1000)) {
  set.seed(42)
  k <- 5
  X <- matrix(runif(n * k, -2, 2), nrow = n, ncol = k)
  colnames(X) <- paste0("x", 1:k)
  lin_d <- rowSums(0.2 * X)
  prob_d <- 1 / (1 + exp(-lin_d))
  d_vec <- as.numeric(runif(n) < prob_d)
  lin_y <- rowSums(0.3 * X)
  y_vec <- 1.0 + 0.5 * d_vec + lin_y + runif(n, -0.5, 0.5)
  save_dgp(data.frame(y = y_vec, d = d_vec, X), sprintf("doubleml_n%d", n))
}

# GARCH
for (n in c(200, 500, 1000)) save_dgp(generate_garch_data(n), sprintf("garch_n%d", n))

# Bivariate (for VAR, VECM, Granger)
for (n in c(100, 200, 500)) save_dgp(generate_bivariate_data(n), sprintf("bivariate_n%d", n))

# Count data (for Poisson, NegBin)
for (n in c(200, 500, 1000)) save_dgp(generate_count_data(n), sprintf("count_n%d", n))

# Zero-inflated data (for ZIP, ZINB, Hurdle)
for (n in c(200, 500, 1000)) save_dgp(generate_zeroinfl_data(n), sprintf("zeroinfl_n%d", n))

# Ordered data (for Ordered Logit)
for (n in c(200, 500, 1000)) save_dgp(generate_ordered_data(n), sprintf("ordered_n%d", n))

# Multinomial data (for Multinomial Logit)
for (n in c(200, 500, 1000)) save_dgp(generate_multinomial_data(n), sprintf("multinomial_n%d", n))

message("Data pre-generation complete.\n")

# ============================================
# REGRESSION BENCHMARKS
# ============================================
message("\n=== Regression ===")

for (n in c(100, 1000, 10000)) {
  d <- generate_regression_data(n)
  save_dgp(d, sprintf("regression_n%d", n))

  r <- run_unified("OLS", "standard", n, 42, "regression", function() {
    lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
  }, extract_lm)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  r <- run_unified("OLS", "HC1", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    coeftest(fit, vcov = vcovHC(fit, type = "HC1"))
  }, extract_lm_hc1)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# PANEL DATA BENCHMARKS
# ============================================
message("\n=== Panel Data ===")

suppressPackageStartupMessages({
  library(plm)
  library(lfe)
})

for (params in list(c(10, 10), c(50, 20), c(100, 50))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  save_dgp(d, sprintf("panel_n%d", n))

  pdata <- pdata.frame(d, index = c("entity", "time"))

  # Fixed Effects (plm)
  r <- run_unified("FixedEffects", "within", n, 42, "panel", function() {
    plm(y ~ x1 + x2, data = pdata, model = "within")
  }, extract_plm)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Random Effects (plm)
  r <- run_unified("RandomEffects", "GLS", n, 42, "panel", function() {
    plm(y ~ x1 + x2, data = pdata, model = "random")
  }, extract_plm)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # HDFE (lfe)
  r <- run_unified("HDFE", "2-way", n, 42, "panel", function() {
    felm(y ~ x1 + x2 | entity + time, data = d)
  }, extract_felm)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Additional Panel Methods: Hausman, Panel_GLS, Arellano_Bond, PVCM, PMG
message("\n=== Additional Panel Methods ===")

# Hausman test
for (params in list(c(10, 10), c(50, 20), c(100, 50))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  save_dgp(d, sprintf("panel_n%d", n))
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("Hausman", "phtest", n, 42, "panel", function() {
    fe <- plm(y ~ x1 + x2, data = local_pdata, model = "within")
    re <- plm(y ~ x1 + x2, data = local_pdata, model = "random")
    phtest(fe, re)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Panel GLS (pggls)
for (params in list(c(10, 10), c(50, 20), c(100, 50))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("Panel_GLS", "pggls", n, 42, "panel", function() {
    pggls(y ~ x1 + x2, data = local_pdata, model = "pooling")
  }, function(fit) {
    s <- summary(fit)
    list(
      coefficients = as.numeric(coef(fit)),
      std_errors = as.numeric(s$coefficients[, "Std. Error"])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Arellano-Bond GMM (pgmm)
for (params in list(c(10, 10), c(50, 20))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("Arellano_Bond", "pgmm", n, 42, "panel", function() {
    pgmm(y ~ lag(y, 1) + x1 + x2 | lag(y, 2:99), data = local_pdata,
          effect = "individual", model = "twosteps")
  }, function(fit) {
    s <- summary(fit)
    list(
      coefficients = as.numeric(coef(fit)),
      std_errors = as.numeric(s$coefficients[, "Std. Error"])
    )
  }, iterations = 10)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# PVCM (random coefficients - Swamy)
for (params in list(c(10, 10), c(50, 20))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("PVCM", "random", n, 42, "panel", function() {
    pvcm(y ~ x1 + x2, data = local_pdata, model = "random")
  }, function(fit) {
    list(
      coefficients = as.numeric(coef(fit))
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# PMG (Mean Group estimator)
for (params in list(c(10, 10), c(50, 20))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("PMG", "pmg", n, 42, "panel", function() {
    pmg(y ~ x1 + x2, data = local_pdata)
  }, function(fit) {
    list(
      coefficients = as.numeric(coef(fit))
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# PANEL: Unit Root Tests
# ============================================
message("\n=== Panel Unit Root ===")

for (params in list(c(20, 10), c(50, 10), c(100, 10))) {
  n_ent <- params[1]
  n_per <- params[2]
  n <- n_ent * n_per

  d <- generate_panel_data(n_ent, n_per)
  pdata <- pdata.frame(d, index = c("entity", "time"))

  local_pdata <- pdata
  r <- run_unified("Panel_Unit_Root", "LLC", n, 42, "panel", function() {
    purtest(local_pdata$y, pmax = 4, test = "levinlin", exo = "intercept")
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic$statistic),
      p_value = as.numeric(fit$statistic$p.value)
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# PANEL: FEGLM (Gaussian)
# ============================================
message("\n=== FEGLM Gaussian ===")

if (requireNamespace("fixest", quietly = TRUE)) {
  suppressPackageStartupMessages(library(fixest))

  for (params in list(c(10, 10), c(50, 10), c(100, 10))) {
    n_ent <- params[1]
    n_per <- params[2]
    n <- n_ent * n_per

    d <- generate_panel_data(n_ent, n_per)
    local_d <- d

    r <- run_unified("FEGLM_Gaussian", "fixest", n, 42, "panel", function() {
      fixest::feglm(y ~ x1 + x2 | entity, data = local_d, family = gaussian())
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        std_errors = as.numeric(se(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: fixest package not installed")
}

# ============================================
# DISCRETE CHOICE BENCHMARKS
# ============================================
message("\n=== Discrete Choice ===")

for (n in c(100, 500, 1000)) {
  d <- generate_binary_data(n)
  save_dgp(d, sprintf("binary_n%d", n))

  r <- run_unified("Logit", "MLE", n, 42, "binary", function() {
    glm(y ~ x1 + x2, data = d, family = binomial(link = "logit"))
  }, extract_glm)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  r <- run_unified("Probit", "MLE", n, 42, "binary", function() {
    glm(y ~ x1 + x2, data = d, family = binomial(link = "probit"))
  }, extract_glm)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# TIME SERIES BENCHMARKS
# ============================================
message("\n=== Time Series ===")

if (requireNamespace("forecast", quietly = TRUE)) {
  suppressPackageStartupMessages(library(forecast))

  for (n in c(100, 200, 500)) {
    d <- generate_time_series(n)
    save_dgp(d, sprintf("timeseries_n%d", n))
    ts_data <- ts(d$y, frequency = 12)

    r <- run_unified("ARIMA", "(1,1,1)", n, 42, "timeseries", function() {
      Arima(ts_data, order = c(1, 1, 1))
    }, extract_arima, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    r <- run_unified("MSTL", "period=12", n, 42, "timeseries", function() {
      mstl(ts_data)
    }, function(res) {
      # ncol(res) includes the Data column; subtract 1 to count only
      # decomposition components (trend + seasonal(s) + remainder)
      list(n_components = ncol(res) - 1)
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: forecast package not installed")
}

# ============================================
# ML BENCHMARKS
# ============================================
message("\n=== Machine Learning ===")

for (n in c(100, 1000, 5000)) {
  d <- generate_cluster_data(n)
  save_dgp(d, sprintf("cluster_n%d", n))
  mat <- as.matrix(d)

  r <- run_unified("K-Means", "k=3", n, 42, "cluster", function() {
    kmeans(mat, centers = 3, nstart = 5, iter.max = 100)
  }, extract_kmeans)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  r <- run_unified("PCA", "k=3", n, 42, "cluster", function() {
    prcomp(mat, center = TRUE, scale. = FALSE)
  }, extract_pca)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# LOESS
# ============================================
message("\n=== LOESS ===")

for (n in c(100, 500, 1000)) {
  d <- generate_regression_data(n, 1)
  save_dgp(d, sprintf("loess_n%d", n))

  r <- run_unified("LOESS", "span=0.75", n, 42, "regression", function() {
    loess(y ~ x1, data = d, span = 0.75, degree = 1)
  }, function(fit) {
    fv <- fit$fitted
    list(
      residual_ss = sum(fit$residuals^2),
      fitted_first = as.numeric(fv[1]),
      fitted_last = as.numeric(fv[length(fv)])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# HIERARCHICAL CLUSTERING
# ============================================
message("\n=== Hierarchical Clustering ===")

for (n in c(100, 500, 1000)) {
  d <- generate_cluster_data(n)
  mat <- as.matrix(d)

  r <- run_unified("Hierarchical", "Ward", n, 42, "cluster", function() {
    hclust(dist(mat), method = "ward.D2")
  }, function(fit) {
    cl <- cutree(fit, k = 3)
    list(
      n_clusters = length(unique(cl)),
      cluster_sizes = as.numeric(sort(table(cl)))
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# RANDOM FOREST
# ============================================
message("\n=== Random Forest ===")

if (requireNamespace("randomForest", quietly = TRUE)) {
  suppressPackageStartupMessages(library(randomForest))

  for (n in c(100, 500, 1000)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)
    target <- mat[, 1]
    features <- mat[, 2:ncol(mat)]

    r <- run_unified("RandomForest", "100trees", n, 42, "cluster", function() {
      randomForest(x = features, y = target, ntree = 100, maxnodes = 10, nodesize = 5)
    }, function(fit) {
      list(mse = mean(fit$mse), rsq = mean(fit$rsq))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: randomForest package not installed")
}

# ============================================
# DBSCAN
# ============================================
message("\n=== DBSCAN ===")

if (requireNamespace("dbscan", quietly = TRUE)) {
  suppressPackageStartupMessages(library(dbscan))

  for (n in c(100, 1000, 5000)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)

    r <- run_unified("DBSCAN", "eps=1.5", n, 42, "cluster", function() {
      dbscan::dbscan(mat, eps = 1.5, minPts = 5)
    }, function(res) {
      list(
        n_clusters = max(res$cluster),
        n_noise = sum(res$cluster == 0),
        cluster_sizes = as.numeric(sort(table(res$cluster[res$cluster > 0])))
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: dbscan package not installed")
}

# ============================================
# FACTOR ANALYSIS
# ============================================
message("\n=== Factor Analysis ===")

# Sign-normalize loadings: for each factor, flip all loadings if the
# loading with the largest absolute value is negative. This resolves
# sign indeterminacy between R and Rust implementations.
sign_normalize_loadings <- function(L) {
  for (j in seq_len(ncol(L))) {
    idx <- which.max(abs(L[, j]))
    if (L[idx, j] < 0) {
      L[, j] <- -L[, j]
    }
  }
  L
}

for (n in c(100, 500, 1000)) {
  d <- generate_factor_data(n)
  save_dgp(d, sprintf("factor_n%d", n))
  mat <- as.matrix(d)

  r <- run_unified("factanal", "none", n, 42, "factor", function() {
    factanal(mat, factors = 3, rotation = "none")
  }, function(fit) {
    L <- sign_normalize_loadings(matrix(fit$loadings, ncol = 3))
    list(
      loadings = as.numeric(L),
      uniquenesses = as.numeric(fit$uniquenesses),
      chi_sq = as.numeric(fit$STATISTIC),
      p_value = as.numeric(fit$PVAL)
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  r <- run_unified("factanal", "varimax", n, 42, "factor", function() {
    factanal(mat, factors = 3, rotation = "varimax")
  }, function(fit) {
    L <- sign_normalize_loadings(matrix(fit$loadings, ncol = 3))
    list(
      loadings = as.numeric(L),
      uniquenesses = as.numeric(fit$uniquenesses),
      chi_sq = as.numeric(fit$STATISTIC),
      p_value = as.numeric(fit$PVAL)
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# FISHER EXACT TEST
# ============================================
message("\n=== Fisher Exact Test ===")

for (n in c(20, 100, 500, 1000)) {
  a <- round(n * 0.3)
  b <- round(n * 0.2)
  cc <- round(n * 0.15)
  dd <- n - a - b - cc
  tab <- matrix(c(a, cc, b, dd), nrow = 2)

  r <- run_unified("Fisher", "twosided", n, 42, "fisher", function() {
    fisher.test(tab, alternative = "two.sided")
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  r <- run_unified("Fisher", "with_ci", n, 42, "fisher", function() {
    fisher.test(tab, alternative = "two.sided", conf.int = TRUE, conf.level = 0.95)
  }, function(res) {
    list(
      statistic = as.numeric(res$estimate),
      p_value = as.numeric(res$p.value),
      conf_int = as.numeric(res$conf.int)
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# ISOTONIC REGRESSION
# ============================================
message("\n=== Isotonic Regression ===")

for (n in c(100, 1000, 10000)) {
  set.seed(42)
  x <- (0:(n - 1)) / n
  y_iso <- x * 2 + runif(n, -0.5, 0.5)

  r <- run_unified("Isotonic_Regression", "PAVA", n, 42, "isotonic", function() {
    isoreg(x, y_iso)
  }, function(fit) {
    list(
      n = length(fit$yf),
      residual_ss = sum((y_iso - fit$yf)^2)
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# JARQUE-BERA
# ============================================
message("\n=== Jarque-Bera ===")

if (requireNamespace("tseries", quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    d <- generate_regression_data(n)
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    resids <- residuals(fit)

    r <- run_unified("Jarque_Bera", "standalone", n, 42, "regression", function() {
      tseries::jarque.bera.test(resids)
    }, extract_test)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: tseries package not installed")
}

# ============================================
# CHANGEPOINT
# ============================================
message("\n=== Changepoint ===")

if (requireNamespace("changepoint", quietly = TRUE)) {
  suppressPackageStartupMessages(library(changepoint))

  for (n in c(100, 500, 1000)) {
    d <- generate_time_series(n)

    r <- run_unified("Changepoint", "PELT", n, 42, "timeseries", function() {
      cpt.mean(d$y, method = "PELT")
    }, function(fit) {
      list(
        n_changepoints = length(cpts(fit)),
        changepoint_locations = as.numeric(cpts(fit))
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: changepoint package not installed")
}

# ============================================
# CAUSAL INFERENCE: DiD
# ============================================
message("\n=== DiD (canonical 2x2) ===")

for (n in c(200, 500, 1000)) {
  d <- generate_did_data(n)
  save_dgp(d, sprintf("did_n%d", n))

  r <- run_unified("DiD", "canonical", n, 42, "did", function() {
    lm(y ~ treatment * post + x1, data = d)
  }, function(fit) {
    s <- summary(fit)
    coefs <- coef(fit)
    ses <- s$coefficients[, "Std. Error"]
    list(
      att = as.numeric(coefs["treatment:post"]),
      se = as.numeric(ses["treatment:post"]),
      coefficients = as.numeric(coefs),
      std_errors = as.numeric(ses)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# CAUSAL INFERENCE: IV/2SLS
# ============================================
message("\n=== IV/2SLS ===")

if (requireNamespace("AER", quietly = TRUE)) {
  suppressPackageStartupMessages(library(AER))

  for (n in c(200, 500, 1000)) {
    d <- generate_iv_data(n)
    save_dgp(d, sprintf("iv_n%d", n))

    r <- run_unified("IV_2SLS", "2sls", n, 42, "iv", function() {
      ivreg(y ~ x_exog + x_endog | x_exog + instrument, data = d)
    }, function(fit) {
      s <- summary(fit)
      list(
        coefficients = as.numeric(coef(fit)),
        std_errors = as.numeric(s$coefficients[, "Std. Error"])
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: AER package not installed")
}

# ============================================
# CAUSAL INFERENCE: RD
# ============================================
message("\n=== RD (sharp) ===")

if (requireNamespace("rdrobust", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_rd_data(n)
    save_dgp(d, sprintf("rd_n%d", n))

    r <- run_unified("RD", "sharp", n, 42, "rd", function() {
      rdrobust::rdrobust(d$y, d$running, c = 0)
    }, function(fit) {
      list(
        tau = as.numeric(fit$coef[1]),
        se = as.numeric(fit$se[1]),
        bandwidth = as.numeric(fit$bws[1, 1])
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: rdrobust package not installed")
}

# ============================================
# CAUSAL INFERENCE: Staggered DiD
# ============================================
message("\n=== Staggered DiD ===")

if (requireNamespace("did", quietly = TRUE)) {
  suppressPackageStartupMessages(library(did))

  for (params in list(c(20, 10), c(50, 10))) {
    n_units <- params[1]
    n_periods <- params[2]
    n <- n_units * n_periods

    d <- generate_staggered_panel(n_units, n_periods)
    save_dgp(d, sprintf("staggered_n%d", n))

    local_d <- d
    r <- run_unified("Staggered_DiD", "CS", n, 42, "staggered_panel", function() {
      att_gt(yname = "y", tname = "time", idname = "unit", gname = "treat_time",
             data = local_d, control_group = "nevertreated", bstrap = FALSE)
    }, function(fit) {
      agg <- did::aggte(fit, type = "simple")
      list(
        overall_att = agg$overall.att,
        overall_se = agg$overall.se,
        att = as.numeric(fit$att),
        se = as.numeric(fit$se)
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: did package not installed")
}

# ============================================
# CAUSAL INFERENCE: ETWFE
# ============================================
message("\n=== ETWFE ===")

# ETWFE is typically done via fixest or manual interaction; use fixest if available
if (requireNamespace("fixest", quietly = TRUE)) {
  suppressPackageStartupMessages(library(fixest))

  for (params in list(c(20, 10), c(50, 10))) {
    n_units <- params[1]
    n_periods <- params[2]
    n <- n_units * n_periods

    d <- generate_staggered_panel(n_units, n_periods)
    local_d <- d
    local_d$unit_f <- factor(local_d$unit)
    local_d$time_f <- factor(local_d$time)

    r <- run_unified("ETWFE", "Wooldridge", n, 42, "staggered_panel", function() {
      feols(y ~ treated | unit_f + time_f, data = local_d)
    }, function(fit) {
      list(
        att = as.numeric(coef(fit)["treated"]),
        se = as.numeric(se(fit)["treated"]),
        coefficients = as.numeric(coef(fit)),
        std_errors = as.numeric(se(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: fixest package not installed")
}

# ============================================
# CAUSAL INFERENCE: Bacon decomposition
# ============================================
message("\n=== Bacon Decomposition ===")

if (requireNamespace("bacondecomp", quietly = TRUE)) {
  for (params in list(c(20, 10), c(50, 10))) {
    n_units <- params[1]
    n_periods <- params[2]
    n <- n_units * n_periods

    d <- generate_staggered_panel(n_units, n_periods)
    local_d <- d

    r <- run_unified("Bacon", "decomp", n, 42, "staggered_panel", function() {
      bacondecomp::bacon(y ~ treated, data = local_d, id_var = "unit", time_var = "time")
    }, function(fit) {
      list(
        n_components = nrow(fit),
        weighted_estimate = sum(fit$estimate * fit$weight)
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: bacondecomp package not installed")
}

# ============================================
# CAUSAL INFERENCE: GSynth
# ============================================
message("\n=== GSynth ===")

if (requireNamespace("gsynth", quietly = TRUE)) {
  suppressPackageStartupMessages(library(gsynth))

  for (params in list(c(20, 10), c(50, 10))) {
    n_units <- params[1]
    n_periods <- params[2]
    n <- n_units * n_periods

    d <- generate_staggered_panel(n_units, n_periods)
    save_dgp(d, sprintf("staggered_n%d", n))

    local_d <- d
    r <- run_unified("GSynth", "gsynth", n, 42, "staggered_panel", function() {
      gsynth(y ~ treated, data = local_d, index = c("unit", "time"),
             force = "two-way", se = FALSE)
    }, function(fit) {
      list(
        att = fit$att.avg,
        se = if (!is.null(fit$se.att)) fit$se.att else NA_real_
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: gsynth package not installed")
}

# ============================================
# CAUSAL INFERENCE: TMLE
# ============================================
message("\n=== TMLE ===")

if (requireNamespace("tmle", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    save_dgp(d, sprintf("treatment_n%d", n))
    W <- as.matrix(d[, c("x1", "x2")])
    local_d <- d
    local_W <- W

    r <- run_unified("TMLE", "ATE", n, 42, "treatment", function() {
      tmle::tmle(Y = local_d$y, A = local_d$treatment, W = local_W)
    }, function(fit) {
      list(
        ate = fit$estimates$ATE$psi,
        se = sqrt(fit$estimates$ATE$var.psi),
        ci_lower = fit$estimates$ATE$CI[1],
        ci_upper = fit$estimates$ATE$CI[2]
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: tmle package not installed")
}

# ============================================
# CAUSAL INFERENCE: CTMLE
# ============================================
message("\n=== CTMLE ===")

if (requireNamespace("ctmle", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    W <- as.matrix(d[, c("x1", "x2")])
    local_d <- d
    local_W <- W

    r <- run_unified("CTMLE", "general", n, 42, "treatment", function() {
      ctmle::ctmleGeneral(Y = local_d$y, A = local_d$treatment, W = local_W,
                          preOrder = 1:ncol(local_W))
    }, function(fit) {
      list(
        ate = as.numeric(fit$est),
        se = as.numeric(fit$se)
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else if (requireNamespace("tmle", quietly = TRUE)) {
  message("  ctmle not installed, using TMLE as proxy for CTMLE")
  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    W <- as.matrix(d[, c("x1", "x2")])
    local_d <- d
    local_W <- W

    r <- run_unified("CTMLE", "tmle_proxy", n, 42, "treatment", function() {
      tmle::tmle(Y = local_d$y, A = local_d$treatment, W = local_W)
    }, function(fit) {
      list(
        ate = fit$estimates$ATE$psi,
        se = sqrt(fit$estimates$ATE$var.psi)
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: ctmle and tmle packages not installed")
}

# ============================================
# CAUSAL INFERENCE: IPW
# ============================================
message("\n=== IPW ===")

for (n in c(200, 500, 1000)) {
  d <- generate_treatment_data(n)
  local_d <- d

  r <- run_unified("IPW", "ATE", n, 42, "treatment", function() {
    ps_fit <- glm(treatment ~ x1 + x2, data = local_d, family = binomial)
    ps <- fitted(ps_fit)
    w1 <- local_d$treatment / ps
    w0 <- (1 - local_d$treatment) / (1 - ps)
    ate <- mean(w1 * local_d$y) - mean(w0 * local_d$y)
    # Bootstrap SE
    n_boot <- 200
    boot_ates <- numeric(n_boot)
    for (b in 1:n_boot) {
      idx <- sample(nrow(local_d), replace = TRUE)
      bd <- local_d[idx, ]
      bps_fit <- glm(treatment ~ x1 + x2, data = bd, family = binomial)
      bps <- fitted(bps_fit)
      bw1 <- bd$treatment / pmax(bps, 0.01)
      bw0 <- (1 - bd$treatment) / pmax(1 - bps, 0.01)
      boot_ates[b] <- mean(bw1 * bd$y) - mean(bw0 * bd$y)
    }
    list(ate = ate, se = sd(boot_ates))
  }, function(res) {
    list(ate = res$ate, se = res$se)
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# CAUSAL INFERENCE: CBPS
# ============================================
message("\n=== CBPS ===")

if (requireNamespace("CBPS", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    local_d <- d

    r <- run_unified("CBPS", "exact", n, 42, "treatment", function() {
      CBPS::CBPS(treatment ~ x1 + x2, data = local_d, ATT = FALSE)
    }, function(fit) {
      # Compute ATE from CBPS weights (Horvitz-Thompson estimator)
      # This matches Rust which also computes ATE from weights
      w <- fit$weights
      treated <- local_d$treatment == 1
      w1 <- w[treated]; w0 <- w[!treated]
      y1 <- local_d$y[treated]; y0 <- local_d$y[!treated]
      ate <- sum(w1 * y1) / sum(w1) - sum(w0 * y0) / sum(w0)
      list(
        ate = ate,
        converged = 1
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: CBPS package not installed")
}

# ============================================
# CAUSAL INFERENCE: Matching (MatchIt)
# ============================================
message("\n=== Matching ===")

if (requireNamespace("MatchIt", quietly = TRUE)) {
  suppressPackageStartupMessages(library(MatchIt))

  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    local_d <- d

    r <- run_unified("Matching", "nearest", n, 42, "treatment", function() {
      matchit(treatment ~ x1 + x2, data = local_d, method = "nearest")
    }, function(fit) {
      list(
        n_matched = sum(fit$weights > 0),
        method = fit$method
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: MatchIt package not installed")
}

# ============================================
# CAUSAL INFERENCE: WeightIt
# ============================================
message("\n=== WeightIt ===")

if (requireNamespace("WeightIt", quietly = TRUE)) {
  suppressPackageStartupMessages(library(WeightIt))

  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    local_d <- d

    r <- run_unified("WeightIt", "logistic", n, 42, "treatment", function() {
      weightit(treatment ~ x1 + x2, data = local_d, method = "ps")
    }, function(fit) {
      list(
        n_weights = length(fit$weights),
        weights_summary = as.numeric(summary(fit$weights))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: WeightIt package not installed")
}

# ============================================
# CAUSAL INFERENCE: Doubly Robust (AIPW)
# ============================================
message("\n=== Doubly Robust ===")

for (n in c(200, 500, 1000)) {
  d <- generate_binary_data(n)
  local_d <- d

  r <- run_unified("Doubly_Robust", "AIPW", n, 42, "binary", function() {
    # Manual AIPW implementation
    ps_fit <- glm(y ~ x1 + x2, data = local_d, family = binomial)
    ps <- fitted(ps_fit)
    # Outcome models
    mu1_fit <- lm(x1 ~ x2, data = local_d[local_d$y == 1, ])
    mu0_fit <- lm(x1 ~ x2, data = local_d[local_d$y == 0, ])
    mu1 <- predict(mu1_fit, newdata = local_d)
    mu0 <- predict(mu0_fit, newdata = local_d)
    aipw <- mean(mu1 - mu0 + local_d$y * (local_d$x1 - mu1) / ps -
                   (1 - local_d$y) * (local_d$x1 - mu0) / (1 - ps))
    # Influence function for SE
    phi <- mu1 - mu0 + local_d$y * (local_d$x1 - mu1) / ps -
           (1 - local_d$y) * (local_d$x1 - mu0) / (1 - ps)
    se <- sd(phi) / sqrt(length(phi))
    list(ate = aipw, se = se)
  }, function(res) {
    list(ate = res$ate, se = res$se)
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# CAUSAL INFERENCE: DoubleML
# ============================================
message("\n=== DoubleML ===")

if (requireNamespace("DoubleML", quietly = TRUE) &&
    requireNamespace("mlr3", quietly = TRUE) &&
    requireNamespace("mlr3learners", quietly = TRUE)) {
  suppressPackageStartupMessages({
    library(DoubleML)
    library(mlr3)
    library(mlr3learners)
  })

  for (n in c(200, 500, 1000)) {
    set.seed(42)
    k <- 5
    X <- matrix(runif(n * k, -2, 2), nrow = n, ncol = k)
    colnames(X) <- paste0("x", 1:k)
    lin_d <- rowSums(0.2 * X)
    prob_d <- 1 / (1 + exp(-lin_d))
    d_vec <- as.numeric(runif(n) < prob_d)
    lin_y <- rowSums(0.3 * X)
    y_vec <- 1.0 + 0.5 * d_vec + lin_y + runif(n, -0.5, 0.5)

    dml_data <- data.frame(y = y_vec, d = d_vec, X)
    save_dgp(dml_data, sprintf("doubleml_n%d", n))

    local_dml_data <- dml_data
    r <- run_unified("DoubleML", "PLR", n, 42, "doubleml", function() {
      obj_dml_data <- DoubleMLData$new(local_dml_data, y_col = "y", d_cols = "d")
      ml_l <- lrn("regr.lm")
      ml_m <- lrn("classif.log_reg", predict_type = "prob")
      dml_plr <- DoubleMLPLR$new(obj_dml_data, ml_l, ml_m)
      dml_plr$fit()
      dml_plr
    }, function(fit) {
      list(
        coefficient = fit$coef,
        se = fit$se
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: DoubleML/mlr3 packages not installed")
}

# ============================================
# CAUSAL INFERENCE: Mediation
# ============================================
message("\n=== Mediation ===")

if (requireNamespace("mediation", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_mediation_data(n)
    save_dgp(d, sprintf("mediation_n%d", n))
    local_d <- d

    r <- run_unified("Mediation", "IPW", n, 42, "mediation", function() {
      med_model <- lm(mediator ~ treatment + x1, data = local_d)
      out_model <- lm(y ~ treatment + mediator + x1, data = local_d)
      mediation::mediate(med_model, out_model, treat = "treatment", mediator = "mediator", sims = 200)
    }, function(fit) {
      list(
        acme = fit$d0,
        ade = fit$z0,
        total = fit$tau.coef,
        prop_mediated = fit$n0
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: mediation package not installed")
}

# ============================================
# CAUSAL INFERENCE: LTMLE
# ============================================
message("\n=== LTMLE ===")

if (requireNamespace("ltmle", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    set.seed(42)
    L1_1 <- runif(n, -1, 1); L1_2 <- runif(n, -1, 1)
    A1 <- as.numeric(runif(n) < 1 / (1 + exp(-0.3 * L1_1)))
    L2_1 <- runif(n, -1, 1); L2_2 <- runif(n, -1, 1)
    A2 <- as.numeric(runif(n) < 1 / (1 + exp(-0.3 * L2_1 - 0.2 * A1)))
    Y <- 1.0 + 0.5 * A1 + 0.3 * A2 + 0.2 * L1_1 + runif(n, -0.5, 0.5)

    ltmle_df <- data.frame(L1_1 = L1_1, L1_2 = L1_2, A1 = A1,
                           L2_1 = L2_1, L2_2 = L2_2, A2 = A2, Y = Y)

    local_ltmle_df <- ltmle_df
    r <- run_unified("LTMLE", "2-period", n, 42, "ltmle", function() {
      ltmle::ltmle(local_ltmle_df,
                   Anodes = c("A1", "A2"),
                   Ynodes = "Y",
                   Lnodes = c("L1_1", "L1_2", "L2_1", "L2_2"),
                   abar = c(1, 1))
    }, function(fit) {
      s <- summary(fit)
      list(estimate = s$treatment$estimate, std_dev = s$treatment$std.dev)
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: ltmle package not installed")
}

# ============================================
# SPATIAL ECONOMETRICS
# ============================================
message("\n=== Spatial Econometrics ===")

if (requireNamespace("spdep", quietly = TRUE) && requireNamespace("spatialreg", quietly = TRUE)) {
  suppressPackageStartupMessages({
    library(spdep)
    library(spatialreg)
  })

  for (n_side in c(10, 20, 32)) {
    n <- n_side * n_side
    d <- generate_spatial_data(n_side)
    save_dgp(d, sprintf("spatial_n%d", n))

    coords <- as.matrix(d[, c("x_coord", "y_coord")])
    nb <- knn2nb(knearneigh(coords, k = 4))
    listw <- nb2listw(nb, style = "W")

    local_d <- d
    local_listw <- listw

    # SAR
    r <- run_unified("SAR", "lagsarlm", n, 42, "spatial", function() {
      lagsarlm(y ~ x, data = local_d, listw = local_listw)
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        rho = fit$rho,
        log_likelihood = as.numeric(logLik(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # SEM
    r <- run_unified("SEM", "errorsarlm", n, 42, "spatial", function() {
      errorsarlm(y ~ x, data = local_d, listw = local_listw)
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        lambda = fit$lambda,
        log_likelihood = as.numeric(logLik(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # SAC (SARAR)
    r <- run_unified("SAC", "sacsarlm", n, 42, "spatial", function() {
      sacsarlm(y ~ x, data = local_d, listw = local_listw)
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        rho = fit$rho,
        lambda = fit$lambda
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # Moran's I test
    r <- run_unified("Moran_Test", "moran.test", n, 42, "spatial", function() {
      moran.test(local_d$y, local_listw, alternative = "greater")
    }, function(fit) {
      list(
        statistic = as.numeric(fit$statistic),
        p_value = as.numeric(fit$p.value)
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # Local Moran (LISA) - skip n_side=32 (too slow for permutations)
    if (n_side <= 20) {
      r <- run_unified("Local_Moran", "localmoran", n, 42, "spatial", function() {
        localmoran(local_d$y, local_listw)
      }, function(fit) {
        n_sig <- sum(fit[, "Pr(z != E(Ii))"] < 0.05, na.rm = TRUE)
        list(n_significant = n_sig)
      }, iterations = 20)
      if (!is.null(r)) results[[length(results) + 1]] <- r
    }
  }
} else {
  message("  Skipping: spdep/spatialreg packages not installed")
}

# ============================================
# SYNTHETIC CONTROL
# ============================================
message("\n=== Synthetic Control ===")

if (requireNamespace("Synth", quietly = TRUE)) {
  for (n_units in c(10, 30)) {
    n_periods <- 10
    n <- n_units * n_periods
    set.seed(42)

    unit_ids <- character(n)
    time_ids <- integer(n)
    outcome <- numeric(n)
    pred1 <- numeric(n)
    pred2 <- numeric(n)

    idx <- 1
    for (u in 0:(n_units - 1)) {
      for (t in 0:(n_periods - 1)) {
        unit_ids[idx] <- sprintf("unit_%d", u)
        time_ids[idx] <- t
        base <- u * 0.5 + t * 0.1
        te <- if (u == 0 && t >= 7) 2.0 else 0.0
        outcome[idx] <- base + te + runif(1, -0.3, 0.3)
        pred1[idx] <- runif(1, 0, 1)
        pred2[idx] <- runif(1, 0, 1)
        idx <- idx + 1
      }
    }

    synth_df <- data.frame(unit = unit_ids, time = time_ids,
                           outcome = outcome, pred1 = pred1, pred2 = pred2,
                           stringsAsFactors = FALSE)
    save_dgp(synth_df, sprintf("synth_%dunits", n_units))

    # Synth package requires numeric unit identifiers
    synth_df$unit_num <- as.numeric(factor(synth_df$unit))
    local_synth <- synth_df

    r <- run_unified("SynthControl", "Nelder-Mead", n, 42, "synth", function() {
      dataprep_out <- Synth::dataprep(
        foo = local_synth,
        predictors = c("pred1", "pred2"),
        predictors.op = "mean",
        dependent = "outcome",
        unit.variable = "unit_num",
        time.variable = "time",
        treatment.identifier = 1,
        controls.identifier = 2:max(local_synth$unit_num),
        time.predictors.prior = 0:6,
        time.optimize.ssr = 0:6,
        time.plot = 0:9
      )
      Synth::synth(data.prep.obj = dataprep_out, optimxmethod = "Nelder-Mead")
    }, function(fit) {
      list(
        loss = fit$loss.v[1],
        solution_w = as.numeric(fit$solution.w)
      )
    }, iterations = 5)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: Synth package not installed")
}

# ============================================
# SURVIVAL ANALYSIS
# ============================================
message("\n=== Survival Analysis ===")

if (requireNamespace("survival", quietly = TRUE)) {
  suppressPackageStartupMessages(library(survival))

  for (n in c(100, 500, 1000)) {
    d <- generate_survival_data(n)
    save_dgp(d, sprintf("survival_n%d", n))
    local_d <- d

    # Kaplan-Meier
    r <- run_unified("KM", "unstratified", n, 42, "survival", function() {
      survfit(Surv(time, event) ~ 1, data = local_d)
    }, extract_survfit)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # Cox PH
    r <- run_unified("CoxPH", "efron", n, 42, "survival", function() {
      coxph(Surv(time, event) ~ x1 + x2, data = local_d, ties = "efron")
    }, extract_coxph)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # Log-rank test
    r <- run_unified("LogRank", "test", n, 42, "survival", function() {
      survdiff(Surv(time, event) ~ group, data = local_d)
    }, function(fit) {
      list(
        chi_sq = fit$chisq,
        p_value = 1 - pchisq(fit$chisq, df = 1)
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: survival package not installed")
}

# ============================================
# SENSITIVITY ANALYSIS: sensemakr
# ============================================
message("\n=== Sensemakr ===")

if (requireNamespace("sensemakr", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_treatment_data(n)
    local_d <- d

    r <- run_unified("Sensemakr", "sensitivity", n, 42, "treatment", function() {
      fit <- lm(y ~ treatment + x1 + x2, data = local_d)
      sensemakr::sensemakr(fit, treatment = "treatment", benchmark_covariates = "x1")
    }, function(fit) {
      list(
        rv_q = fit$sensitivity_stats$rv_q,
        rv_qa = fit$sensitivity_stats$rv_qa
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: sensemakr package not installed")
}

# ============================================
# SENSITIVITY: E-Value
# ============================================
message("\n=== E-Value ===")

if (requireNamespace("EValue", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    d <- generate_regression_data(n)
    local_d <- d

    r <- run_unified("E_Value", "RR", n, 42, "regression", function() {
      # Fit OLS to get an effect estimate, convert to approximate RR
      fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = local_d)
      coef_val <- abs(coef(fit)["x1"])
      # Use exp(beta) as approximate risk ratio
      rr <- exp(coef_val)
      EValue::evalues.RR(rr, lo = exp(coef_val - 1.96 * summary(fit)$coefficients["x1", "Std. Error"]),
                         hi = exp(coef_val + 1.96 * summary(fit)$coefficients["x1", "Std. Error"]))
    }, function(fit) {
      # evalues.RR returns a matrix; row 1 is point estimate, row 2 is CI
      vals <- as.numeric(fit)
      list(
        e_value = vals[1],
        e_value_ci = if (length(vals) >= 2) vals[2] else NA_real_
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: EValue package not installed")
}

# ============================================
# CAUSAL: Marginal Effects
# ============================================
message("\n=== Marginal Effects ===")

if (requireNamespace("margins", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    d <- generate_binary_data(n)
    local_d <- d

    r <- run_unified("Marginal_Effects", "AME_logit", n, 42, "binary", function() {
      fit <- glm(y ~ x1 + x2, data = local_d, family = binomial(link = "logit"))
      margins::margins(fit)
    }, function(fit) {
      s <- summary(fit)
      list(
        ame_x1 = as.numeric(s$AME[s$factor == "x1"]),
        ame_x2 = as.numeric(s$AME[s$factor == "x2"])
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: margins package not installed")
}

# ============================================
# REGRESSION DIAGNOSTICS & VARIANTS
# ============================================
message("\n=== Regression Diagnostics & Variants ===")

# Extractors for diagnostic tests
extract_hc <- function(result) {
  list(
    coefficients = as.numeric(result[, "Estimate"]),
    std_errors = as.numeric(result[, "Std. Error"]),
    r_squared = NA
  )
}

extract_test_stat <- function(result) {
  list(
    statistic = as.numeric(result$statistic),
    p_value = as.numeric(result$p.value)
  )
}

for (n in c(100, 1000, 10000)) {
  d <- generate_regression_data(n)

  # OLS_HC0
  r <- run_unified("OLS_HC0", "robust", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    coeftest(fit, vcov = vcovHC(fit, type = "HC0"))
  }, extract_hc)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # OLS_HC2
  r <- run_unified("OLS_HC2", "robust", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    coeftest(fit, vcov = vcovHC(fit, type = "HC2"))
  }, extract_hc)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # OLS_HC3
  r <- run_unified("OLS_HC3", "robust", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    coeftest(fit, vcov = vcovHC(fit, type = "HC3"))
  }, extract_hc)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # OLS_HAC (Newey-West)
  r <- run_unified("OLS_HAC", "Newey-West", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    coeftest(fit, vcov = vcovHAC(fit))
  }, extract_hc)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Breusch_Godfrey
  r <- run_unified("Breusch_Godfrey", "LM", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    bgtest(fit)
  }, extract_test_stat)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Breusch_Pagan
  r <- run_unified("Breusch_Pagan", "test", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    bptest(fit)
  }, extract_test_stat)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Durbin_Watson
  r <- run_unified("Durbin_Watson", "test", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    dwtest(fit)
  }, function(result) {
    list(statistic = as.numeric(result$statistic))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Harvey_Collier
  r <- run_unified("Harvey_Collier", "test", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    harvtest(fit)
  }, extract_test_stat)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # RESET
  r <- run_unified("RESET", "test", n, 42, "regression", function() {
    fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    resettest(fit)
  }, extract_test_stat)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # VIF
  if (requireNamespace("car", quietly = TRUE)) {
    r <- run_unified("VIF", "diagnostics", n, 42, "regression", function() {
      fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
      car::vif(fit)
    }, function(result) {
      list(vif_values = as.numeric(result))
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }

  # Wald test (drop x1 from full model)
  r <- run_unified("Wald", "F-test", n, 42, "regression", function() {
    fit_full <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = d)
    waldtest(fit_full, . ~ . - x1)
  }, function(result) {
    list(
      statistic = as.numeric(result$F[2]),
      p_value = as.numeric(result$`Pr(>F)`[2])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# OLS_Bootstrap (slower, smaller sample sizes)
if (requireNamespace("boot", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_regression_data(n)

    r <- run_unified("OLS_Bootstrap", "pairs", n, 42, "regression", function() {
      boot::boot(d, function(data, indices) {
        fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = data[indices, ])
        coef(fit)
      }, R = 199)
    }, function(result) {
      list(
        coefficients = as.numeric(result$t0),
        std_errors = as.numeric(apply(result$t, 2, sd))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: boot package not installed")
}

# GLS (AR1)
if (requireNamespace("nlme", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    d <- generate_regression_data(n, k = 1)

    r <- run_unified("GLS", "AR1", n, 42, "regression", function() {
      nlme::gls(y ~ x1, data = d, correlation = nlme::corAR1(0.5, form = ~1))
    }, function(fit) {
      s <- summary(fit)
      list(
        coefficients = as.numeric(coef(fit)),
        std_errors = as.numeric(s$tTable[, "Std.Error"])
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: nlme package not installed")
}

# Quantile Regression
if (requireNamespace("quantreg", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_regression_data(n, k = 3)

    r <- run_unified("Quantile_Regression", "median", n, 42, "regression", function() {
      quantreg::rq(y ~ x1 + x2 + x3, data = d, tau = 0.5)
    }, function(fit) {
      list(coefficients = as.numeric(coef(fit)))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: quantreg package not installed")
}

# Smooth Spline
for (n in c(100, 500, 1000)) {
  d <- generate_regression_data(n, k = 1)

  r <- run_unified("Smooth_Spline", "GCV", n, 42, "regression", function() {
    smooth.spline(d$x1, d$y)
  }, function(fit) {
    list(
      df = fit$df,
      lambda = fit$lambda
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Stepwise Selection
for (n in c(200, 500, 1000)) {
  d <- generate_regression_data(n)

  r <- run_unified("Stepwise", "both_AIC", n, 42, "regression", function() {
    step(lm(y ~ x1 + x2 + x3 + x4 + x5, data = d), direction = "both", trace = 0)
  }, function(fit) {
    list(
      n_selected = length(coef(fit)) - 1,
      aic = AIC(fit)
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# DISCRETE CHOICE: Count Models
# ============================================
message("\n=== Discrete Choice: Count Models ===")

# Poisson GLM
for (n in c(200, 500, 1000)) {
  d <- generate_count_data(n)
  save_dgp(d, sprintf("count_n%d", n))

  r <- run_unified("Poisson", "GLM", n, 42, "count", function() {
    glm(y ~ x1 + x2, data = d, family = poisson)
  }, extract_glm, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Negative Binomial
if (requireNamespace("MASS", quietly = TRUE)) {
  suppressPackageStartupMessages(library(MASS))

  for (n in c(200, 500, 1000)) {
    d <- generate_count_data(n)
    local_d <- d

    r <- run_unified("NegBin", "MLE", n, 42, "count", function() {
      MASS::glm.nb(y ~ x1 + x2, data = local_d)
    }, function(fit) {
      s <- summary(fit)
      list(
        coefficients = as.numeric(coef(fit)),
        std_errors = as.numeric(s$coefficients[, "Std. Error"]),
        theta = fit$theta,
        aic = AIC(fit)
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping NegBin: MASS package not installed")
}

# ZIP, ZINB, Hurdle
if (requireNamespace("pscl", quietly = TRUE)) {
  suppressPackageStartupMessages(library(pscl))

  for (n in c(200, 500, 1000)) {
    d <- generate_zeroinfl_data(n)
    save_dgp(d, sprintf("zeroinfl_n%d", n))
    local_d <- d

    # ZIP
    r <- run_unified("ZIP", "EM", n, 42, "zeroinfl", function() {
      pscl::zeroinfl(y ~ x1 + x2, data = local_d)
    }, function(fit) {
      s <- summary(fit)
      list(
        count_coefficients = as.numeric(s$coefficients$count[, "Estimate"]),
        log_likelihood = as.numeric(logLik(fit)),
        aic = AIC(fit)
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # ZINB
    r <- run_unified("ZINB", "EM", n, 42, "zeroinfl", function() {
      pscl::zeroinfl(y ~ x1 + x2, data = local_d, dist = "negbin")
    }, function(fit) {
      s <- summary(fit)
      list(
        count_coefficients = as.numeric(s$coefficients$count[, "Estimate"]),
        log_likelihood = as.numeric(logLik(fit)),
        theta = fit$theta
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r

    # Hurdle (Poisson)
    r <- run_unified("Hurdle", "Poisson", n, 42, "zeroinfl", function() {
      pscl::hurdle(y ~ x1 + x2, data = local_d)
    }, function(fit) {
      s <- summary(fit)
      list(
        count_coefficients = as.numeric(s$coefficients$count[, "Estimate"]),
        binary_coefficients = as.numeric(s$coefficients$zero[, "Estimate"]),
        log_likelihood = as.numeric(logLik(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping ZIP/ZINB/Hurdle: pscl package not installed")
}

# ============================================
# DISCRETE CHOICE: Ordered & Multinomial
# ============================================
message("\n=== Discrete Choice: Ordered & Multinomial ===")

# Ordered Logit
if (requireNamespace("MASS", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_ordered_data(n)
    save_dgp(d, sprintf("ordered_n%d", n))
    local_d <- d
    local_d$y <- factor(local_d$y, levels = c("1", "2", "3"), ordered = TRUE)

    r <- run_unified("Ordered_Logit", "MLE", n, 42, "ordered", function() {
      MASS::polr(y ~ x1 + x2, data = local_d, method = "logistic")
    }, function(fit) {
      s <- summary(fit)
      list(
        coefficients = as.numeric(fit$coefficients),
        thresholds = as.numeric(fit$zeta),
        log_likelihood = as.numeric(logLik(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping Ordered_Logit: MASS package not installed")
}

# Multinomial Logit
if (requireNamespace("nnet", quietly = TRUE)) {
  suppressPackageStartupMessages(library(nnet))

  for (n in c(200, 500, 1000)) {
    d <- generate_multinomial_data(n)
    save_dgp(d, sprintf("multinomial_n%d", n))
    local_d <- d
    local_d$y <- factor(local_d$y)

    r <- run_unified("Multinomial_Logit", "MLE", n, 42, "multinomial", function() {
      nnet::multinom(y ~ x1 + x2, data = local_d, trace = FALSE)
    }, function(fit) {
      cc <- coef(fit)
      list(
        coefficients = as.numeric(cc),
        log_likelihood = as.numeric(logLik(fit)),
        aic = AIC(fit)
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping Multinomial_Logit: nnet package not installed")
}

# ============================================
# TIME SERIES / FORECASTING EXTENDED
# ============================================

# --- AR ---
message("\n=== AR ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  y_vec <- d$y

  r <- run_unified("AR", "yule_walker", n, 42, "timeseries", function() {
    ar(y_vec, method = "yule-walker", order.max = 5)
  }, function(fit) {
    list(
      coefficients = as.numeric(fit$ar),
      order = fit$order
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- STL ---
message("\n=== STL ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  ts_data <- ts(d$y, frequency = 12)

  r <- run_unified("STL", "period=12", n, 42, "timeseries", function() {
    stl(ts_data, s.window = "periodic")
  }, function(fit) {
    list(n_components = 3)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Decompose ---
message("\n=== Decompose ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  ts_data <- ts(d$y, frequency = 12)

  r <- run_unified("Decompose", "additive", n, 42, "timeseries", function() {
    decompose(ts_data, type = "additive")
  }, function(fit) {
    list(n_components = 3)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Holt-Winters ---
message("\n=== Holt-Winters ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  ts_data <- ts(d$y + 5.0, frequency = 12)

  r <- run_unified("Holt_Winters", "additive", n, 42, "timeseries", function() {
    HoltWinters(ts_data, seasonal = "additive")
  }, function(fit) {
    list(
      alpha = as.numeric(fit$alpha),
      beta = as.numeric(fit$beta),
      gamma = as.numeric(fit$gamma)
    )
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- GARCH ---
message("\n=== GARCH ===")
if (requireNamespace("fGarch", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_garch_data(n)

    r <- run_unified("GARCH", "(1,1)", n, 42, "garch", function() {
      fGarch::garchFit(~ garch(1,1), data = d$y, trace = FALSE)
    }, function(fit) {
      cc <- fit@fit$coef
      list(coefficients = as.numeric(cc[c("omega", "alpha1", "beta1")]))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: fGarch package not installed")
}

# --- Kalman (StructTS level) ---
message("\n=== Kalman ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  ts_data <- ts(d$y, frequency = 1)

  r <- run_unified("Kalman", "local_level", n, 42, "timeseries", function() {
    StructTS(ts_data, type = "level")
  }, function(fit) {
    list(log_likelihood = as.numeric(logLik(fit)))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- StructTS (trend) ---
message("\n=== StructTS ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)
  ts_data <- ts(d$y, frequency = 1)

  r <- run_unified("StructTS", "trend", n, 42, "timeseries", function() {
    StructTS(ts_data, type = "trend")
  }, function(fit) {
    list(coefficients = as.numeric(fit$coef))
  }, iterations = 20)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- VAR ---
message("\n=== VAR ===")
if (requireNamespace("vars", quietly = TRUE)) {
  for (n in c(100, 200, 500)) {
    d <- generate_bivariate_data(n)

    r <- run_unified("VAR", "p=1", n, 42, "bivariate", function() {
      vars::VAR(d, p = 1, type = "const")
    }, function(fit) {
      cc <- sapply(fit$varresult, function(eq) coef(eq))
      list(coefficients = as.numeric(cc))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: vars package not installed")
}

# --- VECM ---
message("\n=== VECM ===")
if (requireNamespace("urca", quietly = TRUE)) {
  for (n in c(200, 500)) {
    d <- generate_bivariate_data(n)

    r <- run_unified("VECM", "rank=1", n, 42, "bivariate", function() {
      urca::ca.jo(d, type = "trace", K = 2, spec = "transitory")
    }, function(fit) {
      list(coefficients = as.numeric(fit@V))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: urca package not installed")
}

# --- Granger ---
message("\n=== Granger ===")
for (n in c(100, 200, 500)) {
  d <- generate_bivariate_data(n)

  r <- run_unified("Granger", "lags=4", n, 42, "bivariate", function() {
    lmtest::grangertest(y1 ~ y2, order = 4, data = d)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$F[2]),
      p_value = as.numeric(fit$`Pr(>F)`[2])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Phillips-Perron ---
message("\n=== Phillips-Perron ===")
if (requireNamespace("tseries", quietly = TRUE)) {
  for (n in c(100, 200, 500)) {
    d <- generate_time_series(n)

    r <- run_unified("Phillips_Perron", "short_lag", n, 42, "timeseries", function() {
      tseries::pp.test(d$y)
    }, function(fit) {
      list(
        statistic = as.numeric(fit$statistic),
        p_value = as.numeric(fit$p.value)
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: tseries package not installed")
}

# --- Box-Ljung ---
message("\n=== Box-Ljung ===")
for (n in c(100, 200, 500)) {
  d <- generate_time_series(n)

  r <- run_unified("Box_Ljung", "lag=10", n, 42, "timeseries", function() {
    Box.test(d$y, lag = 10, type = "Ljung-Box")
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}



# ============================================
# ML: K-Medoids (PAM)
# ============================================
message("\n=== K-Medoids ===")

if (requireNamespace("cluster", quietly = TRUE)) {
  suppressPackageStartupMessages(library(cluster))

  for (n in c(100, 500, 1000)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)
    save_dgp(d, sprintf("cluster_n%d", n))

    local_mat <- mat
    r <- run_unified("K_Medoids", "PAM", n, 42, "cluster", function() {
      cluster::pam(local_mat, k = 3)
    }, function(fit) {
      list(
        n_clusters = length(unique(fit$clustering)),
        total_dissimilarity = fit$objective["swap"],
        avg_silhouette = mean(fit$silinfo$widths[, "sil_width"])
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: cluster package not installed")
}

# ============================================
# ML: SVM (Linear)
# ============================================
message("\n=== SVM ===")

if (requireNamespace("e1071", quietly = TRUE)) {
  suppressPackageStartupMessages(library(e1071))

  for (n in c(200, 500, 1000)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)
    labels <- factor(ifelse(((1:n) - 1) %% 3 == 0, 1, -1))

    local_mat <- mat
    local_labels <- labels
    r <- run_unified("SVM", "linear", n, 42, "cluster", function() {
      e1071::svm(x = local_mat, y = local_labels, kernel = "linear",
                 type = "C-classification", cost = 1.0)
    }, function(fit) {
      list(
        n_support_vectors = sum(fit$nSV),
        converged = fit$convergence == 0
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: e1071 package not installed")
}

# ============================================
# ML: t-SNE
# ============================================
message("\n=== t-SNE ===")

if (requireNamespace("Rtsne", quietly = TRUE)) {
  suppressPackageStartupMessages(library(Rtsne))

  for (n in c(100, 500)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)

    local_mat <- mat
    perp <- min(30, nrow(local_mat) / 3 - 1)
    r <- run_unified("tSNE", "default", n, 42, "cluster", function() {
      Rtsne::Rtsne(local_mat, dims = 2, perplexity = perp,
                   max_iter = 500, check_duplicates = FALSE)
    }, function(fit) {
      list(
        n_components = ncol(fit$Y),
        kl_divergence = tail(fit$itercosts, 1)
      )
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: Rtsne package not installed")
}

# ============================================
# ML: Silhouette
# ============================================
message("\n=== Silhouette ===")

if (requireNamespace("cluster", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    d <- generate_cluster_data(n)
    mat <- as.matrix(d)

    local_mat <- mat
    km_fit <- kmeans(local_mat, centers = 3, nstart = 1, iter.max = 100)
    local_clusters <- km_fit$cluster
    r <- run_unified("Silhouette", "from_kmeans", n, 42, "cluster", function() {
      cluster::silhouette(local_clusters, dist(local_mat))
    }, function(fit) {
      list(
        avg_silhouette = mean(fit[, "sil_width"]),
        n_clusters = length(unique(fit[, "cluster"]))
      )
    }, iterations = 50)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: cluster package not installed")
}

# ============================================
# ML: MDS (cmdscale)
# ============================================
message("\n=== MDS ===")

for (n in c(100, 500)) {
  d <- generate_cluster_data(n)
  mat <- as.matrix(d)

  local_mat <- mat
  r <- run_unified("MDS", "classical", n, 42, "cluster", function() {
    cmdscale(dist(local_mat), k = 2, eig = TRUE)
  }, function(fit) {
    eig_vals <- fit$eig
    pos_eig <- sum(eig_vals[eig_vals > 0])
    abs_eig <- sum(abs(eig_vals))
    list(
      gof_1 = pos_eig / abs_eig,
      gof_2 = sum(eig_vals[1:2]) / abs_eig,
      k = 2
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: ACF
# ============================================
message("\n=== ACF ===")

for (n in c(100, 500, 1000)) {
  d <- generate_time_series(n)
  save_dgp(d, sprintf("timeseries_n%d", n))

  local_y <- d$y
  r <- run_unified("ACF", "correlation", n, 42, "timeseries", function() {
    acf(local_y, lag.max = 20, plot = FALSE)
  }, function(fit) {
    list(
      n_lags = length(fit$lag),
      acf_lag1 = as.numeric(fit$acf[2])
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: PACF
# ============================================
message("\n=== PACF ===")

for (n in c(100, 500, 1000)) {
  d <- generate_time_series(n)

  local_y <- d$y
  r <- run_unified("PACF", "durbin_levinson", n, 42, "timeseries", function() {
    pacf(local_y, lag.max = 20, plot = FALSE)
  }, function(fit) {
    list(
      n_lags = length(fit$lag),
      pacf_lag1 = as.numeric(fit$acf[1])
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: CCF
# ============================================
message("\n=== CCF ===")

for (n in c(100, 200, 500)) {
  set.seed(42)
  t_idx <- 0:(n - 1)
  y1 <- 0.01 * t_idx + sin(t_idx * pi / 6) * 2.0 + runif(n, 0, 0.5)
  y2 <- 0.02 * t_idx + cos(t_idx * pi / 4) * 1.5 + runif(n, 0, 0.5)
  ccf_df <- data.frame(y1 = y1, y2 = y2)
  save_dgp(ccf_df, sprintf("bivariate_ts_n%d", n))

  local_y1 <- y1
  local_y2 <- y2
  r <- run_unified("CCF", "cross_correlation", n, 42, "bivariate_ts", function() {
    ccf(local_y1, local_y2, lag.max = 10, plot = FALSE)
  }, function(fit) {
    lag0_idx <- which(fit$lag == 0)
    list(
      n_lags = length(fit$lag),
      ccf_lag0 = as.numeric(fit$acf[lag0_idx])
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Canonical Correlation
# ============================================
message("\n=== Canonical Correlation ===")

for (n in c(100, 500, 1000)) {
  d <- generate_regression_data(n)

  local_d <- d
  r <- run_unified("Cancor", "canonical", n, 42, "regression", function() {
    cancor(local_d[, c("x1", "x2")], local_d[, c("x3", "x4", "x5")])
  }, function(fit) {
    list(correlations = as.numeric(fit$cor))
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Spline
# ============================================
message("\n=== Spline ===")

for (n in c(100, 500, 1000)) {
  set.seed(42)
  x_sp <- (0:(n - 1)) / n
  y_sp <- sin(x_sp * pi * 2) + runif(n, -0.1, 0.1)
  sp_df <- data.frame(x = x_sp, y = y_sp)
  save_dgp(sp_df, sprintf("spline_n%d", n))

  local_x <- x_sp
  local_y <- y_sp
  n_out <- n * 3
  r <- run_unified("Spline", "natural", n, 42, "spline", function() {
    spline(local_x, local_y, n = n_out, method = "natural")
  }, function(fit) {
    list(
      n_out = length(fit$x),
      y_first = fit$y[1]
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: t-test (one sample)
# ============================================
message("\n=== t-test ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  x_tt <- 5.0 + runif(n, -1, 1)
  tt_df <- data.frame(x = x_tt)
  save_dgp(tt_df, sprintf("normal_n%d", n))

  local_x <- x_tt
  r <- run_unified("t_test", "one_sample", n, 42, "normal", function() {
    t.test(local_x, mu = 5)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value),
      df = as.numeric(fit$parameter)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Wilcoxon signed-rank
# ============================================
message("\n=== Wilcoxon ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  x_wt <- 5.0 + runif(n, -1, 1)

  local_x <- x_wt
  r <- run_unified("Wilcoxon", "signed_rank", n, 42, "normal", function() {
    wilcox.test(local_x, mu = 5)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: KS test (one sample)
# ============================================
message("\n=== KS test ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  x_ks <- 5.0 + runif(n, -1, 1)

  local_x <- x_ks
  r <- run_unified("KS_test", "one_sample", n, 42, "normal", function() {
    ks.test(local_x, "pnorm", mean = 5, sd = 1)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Shapiro-Wilk
# ============================================
message("\n=== Shapiro-Wilk ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  x_sw <- 5.0 + runif(n, -1, 1)

  local_x <- x_sw
  r <- run_unified("Shapiro_Wilk", "test", n, 42, "normal", function() {
    shapiro.test(local_x)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: ANOVA (one-way)
# ============================================
message("\n=== ANOVA ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  n_per_group <- n %/% 3
  groups <- c(rep("g0", n_per_group), rep("g1", n_per_group), rep("g2", n - 2 * n_per_group))
  y_anova <- c(
    5.0 + runif(n_per_group, -1, 1),
    6.0 + runif(n_per_group, -1, 1),
    7.0 + runif(n - 2 * n_per_group, -1, 1)
  )
  anova_df <- data.frame(y = y_anova, group = factor(groups))
  save_dgp(anova_df, sprintf("anova_n%d", n))

  local_df <- anova_df
  r <- run_unified("ANOVA", "one_way", n, 42, "anova", function() {
    summary(aov(y ~ group, data = local_df))
  }, function(fit) {
    list(
      f_statistic = as.numeric(fit[[1]]["group", "F value"]),
      p_value = as.numeric(fit[[1]]["group", "Pr(>F)"])
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: ANOVA Two-Way
# ============================================
message("\n=== ANOVA Two-Way ===")

for (n in c(100, 500, 1000)) {
  set.seed(42)
  factor1 <- factor(rep(c("A", "B"), length.out = n))
  factor2 <- factor(rep(c("X", "Y", "Z"), length.out = n))
  y_tw <- 1.0 + ifelse(factor1 == "B", 0.5, 0) +
          ifelse(factor2 == "Y", 0.3, ifelse(factor2 == "Z", 0.8, 0)) +
          ifelse(factor1 == "B" & factor2 == "Z", 0.4, 0) +
          runif(n, -0.5, 0.5)
  tw_df <- data.frame(y = y_tw, factor1 = factor1, factor2 = factor2)

  local_df <- tw_df
  r <- run_unified("ANOVA_TwoWay", "full", n, 42, "twoway_anova", function() {
    summary(aov(y ~ factor1 * factor2, data = local_df))
  }, function(fit) {
    tab <- fit[[1]]
    list(
      f_statistic_1 = as.numeric(tab["factor1", "F value"]),
      f_statistic_2 = as.numeric(tab["factor2", "F value"]),
      f_interaction = as.numeric(tab["factor1:factor2", "F value"])
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Kruskal-Wallis
# ============================================
message("\n=== Kruskal-Wallis ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  n_per_group <- n %/% 3
  groups <- c(rep("g0", n_per_group), rep("g1", n_per_group), rep("g2", n - 2 * n_per_group))
  y_kw <- c(
    5.0 + runif(n_per_group, -1, 1),
    6.0 + runif(n_per_group, -1, 1),
    7.0 + runif(n - 2 * n_per_group, -1, 1)
  )
  kw_df <- data.frame(y = y_kw, group = factor(groups))

  local_df <- kw_df
  r <- run_unified("Kruskal_Wallis", "test", n, 42, "anova", function() {
    kruskal.test(y ~ group, data = local_df)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Friedman
# ============================================
message("\n=== Friedman ===")

for (n in c(30, 100, 500)) {
  set.seed(42)
  # n blocks x 3 treatments
  mat_friedman <- matrix(c(
    5.0 + runif(n, -1, 1),
    6.0 + runif(n, -1, 1),
    7.0 + runif(n, -1, 1)
  ), nrow = n, ncol = 3)
  colnames(mat_friedman) <- c("t0", "t1", "t2")
  fr_df <- as.data.frame(mat_friedman)
  save_dgp(fr_df, sprintf("friedman_n%d", n))

  local_mat <- mat_friedman
  r <- run_unified("Friedman", "test", n, 42, "friedman", function() {
    friedman.test(local_mat)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Chi-squared (goodness of fit)
# ============================================
message("\n=== Chi-squared ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  observed <- (n / 5) + runif(5, -5, 5)
  observed <- abs(observed)  # ensure positive

  local_obs <- observed
  r <- run_unified("Chi_squared", "gof", n, 42, "chisq", function() {
    chisq.test(local_obs)
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS: Correlation Test
# ============================================
message("\n=== Cor Test ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  x_cor <- runif(n, -2, 2)
  y_cor <- 0.5 * x_cor + runif(n, -0.5, 0.5)
  cor_df <- data.frame(x = x_cor, y = y_cor)
  save_dgp(cor_df, sprintf("bivariate_n%d", n))

  local_x <- x_cor
  local_y <- y_cor
  r <- run_unified("Cor_Test", "pearson", n, 42, "bivariate", function() {
    cor.test(local_x, local_y, method = "pearson")
  }, function(fit) {
    list(
      statistic = as.numeric(fit$statistic),
      p_value = as.numeric(fit$p.value),
      estimate = as.numeric(fit$estimate)
    )
  }, iterations = 50)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# Group 2: Stats B + Survival + Causal
# ============================================

# --- Bartlett ---
message("\n=== Bartlett ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  g1 <- rnorm(n/2, 0, 1); g2 <- rnorm(n/2, 0, 1.5)
  r <- run_unified("Bartlett", "two_group", n, 42, "variance", function() {
    bartlett.test(list(g1, g2))
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Fligner ---
message("\n=== Fligner ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  g1 <- rnorm(n/2, 0, 1); g2 <- rnorm(n/2, 0, 1.5)
  r <- run_unified("Fligner", "two_group", n, 42, "variance", function() {
    fligner.test(list(g1, g2))
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Mood ---
message("\n=== Mood ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n/2, 0, 1); y_val <- rnorm(n/2, 0, 1.5)
  r <- run_unified("Mood", "two_sample", n, 42, "variance", function() {
    mood.test(x, y_val)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Ansari ---
message("\n=== Ansari ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n/2, 0, 1); y_val <- rnorm(n/2, 0, 1.5)
  r <- run_unified("Ansari", "two_sample", n, 42, "variance", function() {
    ansari.test(x, y_val)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- McNemar ---
message("\n=== McNemar ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  a <- sample(0:1, n, replace = TRUE)
  b <- ifelse(runif(n) < 0.7, a, 1 - a)
  tab <- table(factor(a, 0:1), factor(b, 0:1))
  r <- run_unified("McNemar", "paired", n, 42, "paired_binary", function() {
    mcnemar.test(tab)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Mantel_Haenszel ---
message("\n=== Mantel_Haenszel ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  tables <- array(0, dim = c(2, 2, 3))
  for (s in 1:3) {
    ns <- n %/% 3
    a <- sample(0:1, ns, replace = TRUE, prob = c(0.4, 0.6))
    b <- ifelse(runif(ns) < 0.5 + 0.1*s, a, 1 - a)
    tables[,,s] <- table(factor(a, 0:1), factor(b, 0:1))
  }
  r <- run_unified("Mantel_Haenszel", "stratified", n, 42, "stratified", function() {
    mantelhaen.test(tables)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- MANOVA ---
message("\n=== MANOVA ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  group <- factor(rep(c("A","B","C"), length.out = n))
  y1 <- rnorm(n) + ifelse(group == "B", 0.5, ifelse(group == "C", 1.0, 0))
  y2 <- rnorm(n) + ifelse(group == "B", 0.3, ifelse(group == "C", 0.7, 0))
  d <- data.frame(y1, y2, group)
  r <- run_unified("MANOVA", "one_way", n, 42, "multivariate", function() {
    summary(manova(cbind(y1, y2) ~ group, data = d))
  }, function(fit) {
    stats <- fit$stats
    list(
      pillai = as.numeric(stats["group", "Pillai"]),
      wilks = as.numeric(stats["group", "Wilks"])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Median_Polish ---
message("\n=== Median_Polish ===")
for (n in c(25, 100, 400)) {
  set.seed(42)
  side <- as.integer(sqrt(n))
  mat <- matrix(rnorm(side * side), nrow = side)
  r <- run_unified("Median_Polish", "default", n, 42, "matrix", function() {
    medpolish(mat, trace.iter = FALSE)
  }, function(fit) {
    list(grand_median = as.numeric(fit$overall))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Oneway (Welch) ---
message("\n=== Oneway ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  group <- factor(rep(c("A","B","C"), length.out = n))
  values <- rnorm(n) + ifelse(group == "B", 0.5, ifelse(group == "C", 1.0, 0))
  r <- run_unified("Oneway", "welch", n, 42, "grouped", function() {
    oneway.test(values ~ group)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Var_Test ---
message("\n=== Var_Test ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n/2, 0, 1); y_val <- rnorm(n/2, 0, 1.5)
  r <- run_unified("Var_Test", "f_test", n, 42, "variance", function() {
    var.test(x, y_val)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Prop_Test ---
message("\n=== Prop_Test ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- sum(rbinom(n, 1, 0.6))
  r <- run_unified("Prop_Test", "one_sample", n, 42, "binomial", function() {
    prop.test(x, n, p = 0.5)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Prop_Trend ---
message("\n=== Prop_Trend ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  k <- 5
  trials <- rep(n %/% k, k)
  probs <- seq(0.3, 0.7, length.out = k)
  successes <- rbinom(k, trials, probs)
  r <- run_unified("Prop_Trend", "default", n, 42, "trend", function() {
    prop.trend.test(successes, trials)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Binom_Test ---
message("\n=== Binom_Test ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- sum(rbinom(n, 1, 0.6))
  r <- run_unified("Binom_Test", "default", n, 42, "binomial", function() {
    binom.test(x, n, p = 0.5)
  }, function(fit) {
    list(p_value = as.numeric(fit$p.value), estimate = as.numeric(fit$estimate))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Poisson_Test ---
message("\n=== Poisson_Test ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- sum(rpois(n, 5))
  r <- run_unified("Poisson_Test", "default", n, 42, "poisson", function() {
    poisson.test(x, n)
  }, function(fit) {
    list(p_value = as.numeric(fit$p.value), estimate = as.numeric(fit$estimate))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Power_Analysis ---
message("\n=== Power_Analysis ===")
for (n in c(50, 100, 200)) {
  r <- run_unified("Power_Analysis", "t_test", n, 42, "computed", function() {
    power.t.test(n = n, delta = 0.5, sd = 1, sig.level = 0.05)
  }, function(fit) {
    list(power = as.numeric(fit$power))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Pairwise_t ---
message("\n=== Pairwise_t ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  group <- factor(rep(c("A","B","C"), length.out = n))
  values <- rnorm(n) + ifelse(group == "B", 0.5, ifelse(group == "C", 1.0, 0))
  r <- run_unified("Pairwise_t", "bonferroni", n, 42, "grouped", function() {
    pairwise.t.test(values, group, p.adjust.method = "bonferroni")
  }, function(fit) {
    list(p_values = as.numeric(na.omit(as.vector(fit$p.value))))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Pairwise_Wilcox ---
message("\n=== Pairwise_Wilcox ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  group <- factor(rep(c("A","B","C"), length.out = n))
  values <- rnorm(n) + ifelse(group == "B", 0.5, ifelse(group == "C", 1.0, 0))
  r <- run_unified("Pairwise_Wilcox", "bonferroni", n, 42, "grouped", function() {
    pairwise.wilcox.test(values, group, p.adjust.method = "bonferroni")
  }, function(fit) {
    list(p_values = as.numeric(na.omit(as.vector(fit$p.value))))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Quade ---
message("\n=== Quade ===")
for (n in c(30, 100, 200)) {
  set.seed(42)
  k <- 3
  mat <- matrix(rnorm(n * k), nrow = n, ncol = k)
  mat[, 2] <- mat[, 2] + 0.5
  mat[, 3] <- mat[, 3] + 1.0
  r <- run_unified("Quade", "default", n, 42, "blocked", function() {
    quade.test(mat)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Tukey ---
message("\n=== Tukey ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  group <- factor(rep(c("A","B","C"), length.out = n))
  values <- rnorm(n) + ifelse(group == "B", 0.5, ifelse(group == "C", 1.0, 0))
  d <- data.frame(y = values, group = group)
  r <- run_unified("Tukey", "hsd", n, 42, "grouped", function() {
    TukeyHSD(aov(y ~ group, data = d))
  }, function(fit) {
    list(p_values = as.numeric(fit$group[, "p adj"]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Mahalanobis ---
message("\n=== Mahalanobis ===")
for (n in c(100, 500, 1000)) {
  d <- generate_cluster_data(n, k = 3)
  mat <- as.matrix(d)
  center <- colMeans(mat)
  cov_mat <- cov(mat)
  r <- run_unified("Mahalanobis", "default", n, 42, "cluster", function() {
    mahalanobis(mat, center, cov_mat)
  }, function(fit) {
    list(distances = as.numeric(fit[1:min(10, length(fit))]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Robust_Stats ---
message("\n=== Robust_Stats ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n, 5, 1)
  r <- run_unified("Robust_Stats", "default", n, 42, "normal", function() {
    list(fivenum = fivenum(x), iqr = IQR(x))
  }, function(fit) {
    list(fivenum_values = as.numeric(fit$fivenum), iqr = fit$iqr)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Weighted ---
message("\n=== Weighted ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n, 5, 1)
  w <- runif(n, 0.1, 2)
  r <- run_unified("Weighted", "mean", n, 42, "normal", function() {
    list(mean = weighted.mean(x, w))
  }, function(fit) {
    list(mean = fit$mean)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- Loglin ---
message("\n=== Loglin ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  tab <- array(rpois(8, n/8) + 1, dim = c(2, 2, 2))
  r <- run_unified("Loglin", "default", n, 42, "contingency", function() {
    loglin(tab, margin = list(c(1,2), c(2,3)))
  }, function(fit) {
    list(deviance = as.numeric(fit$lrt))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- P_Adjust ---
message("\n=== P_Adjust ===")
for (n in c(10, 50, 100)) {
  set.seed(42)
  pvals <- runif(n, 0, 1)
  r <- run_unified("P_Adjust", "bonferroni", n, 42, "pvalues", function() {
    p.adjust(pvals, method = "bonferroni")
  }, function(fit) {
    list(adjusted_p_values = as.numeric(fit[1:min(10, length(fit))]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- RD_Multi ---
message("\n=== RD_Multi ===")
if (requireNamespace("rdrobust", quietly = TRUE)) {
  for (n in c(500, 1000)) {
    set.seed(42)
    cutoffs <- c(-0.5, 0.0, 0.5)
    running <- runif(n, -1.5, 1.5)
    y <- 0.5 + 0.3 * running
    for (c_val in cutoffs) y <- y + ifelse(running >= c_val, 0.5, 0)
    y <- y + rnorm(n, 0, 0.5)
    d <- data.frame(y = y, running = running)
    save_dgp(d, sprintf("rdmulti_n%d", n))
    r <- run_unified("RD_Multi", "multi_cutoff", n, 42, "rdmulti", function() {
      lapply(cutoffs, function(c_val) rdrobust::rdrobust(d$y, d$running, c = c_val))
    }, function(fits) {
      list(coefficients = sapply(fits, function(f) as.numeric(f$coef[1])))
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: rdrobust package not installed")
}

# --- AFT ---
message("\n=== AFT ===")
if (requireNamespace("survival", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    d <- generate_survival_data(n)
    save_dgp(d, sprintf("survival_n%d", n))
    r <- run_unified("AFT", "weibull", n, 42, "survival", function() {
      survival::survreg(survival::Surv(time, event) ~ x1 + x2, data = d, dist = "weibull")
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        log_likelihood = as.numeric(fit$loglik[2])
      )
    })
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: survival package not installed")
}

# --- Competing_Risks ---
message("\n=== Competing_Risks ===")
if (requireNamespace("cmprsk", quietly = TRUE)) {
  for (n in c(200, 500, 1000)) {
    set.seed(42)
    time <- rexp(n, 0.1)
    event_type <- sample(0:2, n, replace = TRUE, prob = c(0.3, 0.4, 0.3))
    d <- data.frame(time = time, event_type = event_type)
    save_dgp(d, sprintf("comprisk_n%d", n))
    group <- factor(rep(c(1,2), length.out = n))
    r <- run_unified("Competing_Risks", "cuminc", n, 42, "comprisk", function() {
      cmprsk::cuminc(d$time, d$event_type, group)
    }, function(fit) {
      list(n_events = length(unique(d$event_type)) - 1)
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: cmprsk package not installed")
}

# ============================================
# Group 3: Stragglers
# ============================================

# --- OLS_Clustered ---
message("\n=== OLS_Clustered ===")
for (n in c(100, 500, 1000)) {
  d <- generate_panel_data(n %/% 10, 10)
  save_dgp(d, sprintf("panel_%d_10", n %/% 10))
  r <- run_unified("OLS_Clustered", "entity", nrow(d), 42, "panel", function() {
    fit <- lm(y ~ x1 + x2, data = d)
    lmtest::coeftest(fit, vcov = sandwich::vcovCL(fit, cluster = d$entity))
  }, extract_lm_hc1)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- OLS_Driscoll_Kraay ---
message("\n=== OLS_Driscoll_Kraay ===")
for (n in c(100, 500, 1000)) {
  d <- generate_panel_data(n %/% 10, 10)
  save_dgp(d, sprintf("panel_%d_10", n %/% 10))
  r <- run_unified("OLS_Driscoll_Kraay", "entity", nrow(d), 42, "panel", function() {
    fit <- lm(y ~ x1 + x2, data = d)
    lmtest::coeftest(fit, vcov = sandwich::vcovSCC(fit, cluster = d$entity))
  }, extract_lm_hc1)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# --- NLS ---
message("\n=== NLS ===")
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x <- seq(0.1, 10, length.out = n)
  y <- 2 * exp(-0.3 * x) + rnorm(n, 0, 0.1)
  d <- data.frame(y = y, x = x)
  save_dgp(d, sprintf("nls_n%d", n))
  r <- run_unified("NLS", "exp_decay", n, 42, "nls", function() {
    nls(y ~ a * exp(-b * x), data = d, start = list(a = 1, b = 0.1))
  }, function(fit) {
    list(
      coefficients = as.numeric(coef(fit)),
      residual_sse = as.numeric(sum(residuals(fit)^2))
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}


# ============================================
# STATS B: Variance/Scale Tests
# ============================================
message("\n=== Variance/Scale Tests ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1)
  g2 <- runif(n, -0.5, 1.5)
  g3 <- runif(n, -2, 2)

  # Bartlett
  local_groups <- list(g1=g1, g2=g2, g3=g3)
  r <- run_unified("Bartlett", "test", n, 42, "groups", function() {
    bartlett.test(local_groups)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Fligner
  r <- run_unified("Fligner", "test", n, 42, "groups", function() {
    fligner.test(local_groups)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Mood
  r <- run_unified("Mood", "test", n, 42, "groups", function() {
    mood.test(g1, g2)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Ansari
  r <- run_unified("Ansari", "test", n, 42, "groups", function() {
    ansari.test(g1, g2)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Var_Test
  r <- run_unified("Var_Test", "F-test", n, 42, "groups", function() {
    var.test(g1, g2)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Oneway (Welch)
  local_df <- data.frame(
    y = c(g1, g2, g3),
    group = factor(rep(c("g1","g2","g3"), each=n))
  )
  r <- run_unified("Oneway", "Welch", n, 42, "groups", function() {
    oneway.test(y ~ group, data = local_df)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS B: Categorical Tests
# ============================================
message("\n=== Categorical Tests ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  a <- as.integer(n * 0.4)
  b <- as.integer(n * 0.15) + sample(1:4, 1)
  cc <- as.integer(n * 0.25) + sample(1:4, 1)
  d <- n - a - b - cc
  local_mat <- matrix(c(a, cc, b, d), nrow=2)  # R fills by column

  r <- run_unified("McNemar", "test", n, 42, "table_2x2", function() {
    mcnemar.test(local_mat)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Mantel-Haenszel
for (n in c(50, 200)) {
  set.seed(42)
  n_strata <- n %/% 25
  arr <- array(0, dim=c(2, 2, n_strata))
  for (k in 1:n_strata) {
    arr[1,1,k] <- runif(1, 5, 20)
    arr[1,2,k] <- runif(1, 3, 15)
    arr[2,1,k] <- runif(1, 3, 15)
    arr[2,2,k] <- runif(1, 5, 20)
  }
  local_arr <- arr
  r <- run_unified("Mantel_Haenszel", "CMH", n, 42, "stratified_2x2", function() {
    mantelhaen.test(local_arr)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS B: Multivariate
# ============================================
message("\n=== Multivariate Tests ===")

for (n in c(50, 200, 1000)) {
  set.seed(42)
  n_groups <- 3
  n_per <- n %/% n_groups
  y1 <- numeric(n)
  y2 <- numeric(n)
  group <- character(n)
  for (g in 0:(n_groups-1)) {
    for (i in 1:n_per) {
      row <- g * n_per + i
      y1[row] <- g * 1.5 + runif(1, -1, 1)
      y2[row] <- g * 0.8 + runif(1, -1, 1)
      group[row] <- paste0("g", g)
    }
  }
  # Fill remainder
  if (n_groups * n_per < n) {
    for (row in (n_groups*n_per+1):n) {
      y1[row] <- runif(1, -1, 1)
      y2[row] <- runif(1, -1, 1)
      group[row] <- "g0"
    }
  }
  local_df <- data.frame(y1=y1, y2=y2, group=factor(group))

  r <- run_unified("MANOVA", "one-way", n, 42, "multivariate_groups", function() {
    summary(manova(cbind(y1, y2) ~ group, data = local_df))
  }, function(fit) {
    stats <- fit$stats
    list(
      pillai = as.numeric(stats[1, "Pillai"]),
      wilks = as.numeric(stats[1, "Wilks"])
    )
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Median_Polish
for (n in c(50, 200)) {
  set.seed(42)
  rows <- as.integer(sqrt(n))
  cols <- rows
  mat <- matrix(0, nrow=rows, ncol=cols)
  for (r in 1:rows) {
    for (c_idx in 1:cols) {
      mat[r, c_idx] <- (r-1) * 0.5 + (c_idx-1) * 0.3 + runif(1, -0.5, 0.5)
    }
  }
  local_mat <- mat

  r <- run_unified("Median_Polish", "iterative", n, 42, "matrix", function() {
    medpolish(local_mat)
  }, function(fit) {
    list(grand_median = fit$overall)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS B: Proportion/Binomial/Poisson Tests
# ============================================
message("\n=== Proportion/Count Tests ===")

for (n in c(50, 200, 1000)) {
  x <- as.integer(n * 0.55)

  r <- run_unified("Prop_Test", "one-sample", n, 42, "proportion", function() {
    prop.test(x, n, p = 0.5)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Prop_Trend
for (n in c(50, 200)) {
  k <- 5
  x_vals <- sapply(0:(k-1), function(i) (n %/% k) %/% 2 + i * 3)
  n_vals <- rep(n %/% k, k)

  r <- run_unified("Prop_Trend", "test", n, 42, "proportion_trend", function() {
    prop.trend.test(x_vals, n_vals)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Binom_Test
for (n in c(50, 200, 1000)) {
  x <- as.integer(n * 0.55)

  r <- run_unified("Binom_Test", "exact", n, 42, "binomial", function() {
    binom.test(x, n, p = 0.5)
  }, function(fit) {
    list(p_value = fit$p.value, estimate = as.numeric(fit$estimate))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Poisson_Test
for (n in c(50, 200, 1000)) {
  r <- run_unified("Poisson_Test", "exact", n, 42, "poisson", function() {
    poisson.test(n, T = 1, r = n)
  }, function(fit) {
    list(p_value = fit$p.value, estimate = as.numeric(fit$estimate))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Power_Analysis
for (n in c(50, 200, 1000)) {
  r <- run_unified("Power_Analysis", "t-test", n, 42, "power", function() {
    power.t.test(n = n, delta = 0.5, sd = 1, sig.level = 0.05, type = "two.sample")
  }, function(fit) {
    list(power = fit$power)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS B: Multiple Comparisons
# ============================================
message("\n=== Multiple Comparisons ===")

for (n in c(100, 500)) {
  set.seed(42)
  n_per <- n %/% 3
  values <- numeric(n)
  group_labels <- character(n)
  for (g in 0:2) {
    offset <- g * 0.5
    for (i in 1:n_per) {
      idx <- g * n_per + i
      values[idx] <- offset + runif(1, -1, 1)
      group_labels[idx] <- paste0("g", g)
    }
  }
  if (3 * n_per < n) {
    for (idx in (3*n_per+1):n) {
      values[idx] <- runif(1, -1, 1)
      group_labels[idx] <- "g0"
    }
  }
  local_values <- values
  local_groups <- factor(group_labels)

  # Pairwise_t
  r <- run_unified("Pairwise_t", "Holm", n, 42, "groups", function() {
    pairwise.t.test(local_values, local_groups, p.adjust.method = "holm")
  }, function(fit) {
    list(p_values = as.numeric(na.omit(as.vector(fit$p.value))))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Pairwise_Wilcox
  r <- run_unified("Pairwise_Wilcox", "Holm", n, 42, "groups", function() {
    pairwise.wilcox.test(local_values, local_groups, p.adjust.method = "holm")
  }, function(fit) {
    list(p_values = as.numeric(na.omit(as.vector(fit$p.value))))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Tukey HSD
  local_df <- data.frame(y = local_values, group = local_groups)
  r <- run_unified("Tukey", "HSD", n, 42, "groups", function() {
    TukeyHSD(aov(y ~ group, data = local_df))
  }, function(fit) {
    list(p_values = as.numeric(fit$group[, "p adj"]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Quade test
for (n in c(30, 100)) {
  set.seed(42)
  n_blocks <- n
  n_treatments <- 3
  mat <- matrix(0, nrow=n_blocks, ncol=n_treatments)
  for (i in 1:n_blocks) {
    for (t in 1:n_treatments) {
      mat[i, t] <- (t-1) * 0.3 + runif(1, -1, 1)
    }
  }
  local_df <- data.frame(
    y = as.vector(mat),
    block = factor(rep(1:n_blocks, times=n_treatments)),
    treatment = factor(rep(paste0("t", 0:(n_treatments-1)), each=n_blocks))
  )

  r <- run_unified("Quade", "test", n, 42, "blocked", function() {
    quade.test(y ~ treatment | block, data = local_df)
  }, extract_test)
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# P_Adjust
for (n in c(50, 200, 1000)) {
  set.seed(42)
  pvals <- runif(n, 0, 1)

  r <- run_unified("P_Adjust", "BH", n, 42, "p_values", function() {
    p.adjust(pvals, method = "BH")
  }, function(result) {
    list(adjusted_p_values = as.numeric(result[1:min(3, length(result))]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# STATS B: Utility / Robust / Weighted
# ============================================
message("\n=== Utility Stats ===")

for (n in c(100, 1000)) {
  set.seed(42)
  data <- runif(n, -5, 5)
  weights <- runif(n, 0.1, 2)

  # Robust_Stats
  local_data <- data
  r <- run_unified("Robust_Stats", "fivenum+iqr", n, 42, "random", function() {
    fn <- fivenum(local_data)
    iq <- IQR(local_data)
    list(fn = fn, iq = iq)
  }, function(result) {
    list(fivenum = as.numeric(result$fn), iqr = result$iq)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Weighted_Mean
  local_w <- weights
  r <- run_unified("Weighted", "mean", n, 42, "random", function() {
    weighted.mean(local_data, local_w)
  }, function(result) {
    list(mean = as.numeric(result))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r

  # Mahalanobis
  set.seed(42)
  mdata <- matrix(runif(n * 3, -2, 2), nrow=n, ncol=3)
  local_mdata <- mdata
  r <- run_unified("Mahalanobis", "distances", n, 42, "multivariate", function() {
    mahalanobis(local_mdata, colMeans(local_mdata), cov(local_mdata))
  }, function(result) {
    list(distances_first3 = as.numeric(result[1:min(3, length(result))]))
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# Loglin (3-way table)
for (n in c(50, 200)) {
  set.seed(42)
  tab <- array(runif(8, 5, 50), dim=c(2, 2, 2))
  local_tab <- tab

  r <- run_unified("Loglin", "3-way", n, 42, "contingency", function() {
    loglin(local_tab, margin = list(1:2, 2:3))
  }, function(fit) {
    list(deviance = fit$lrt)
  })
  if (!is.null(r)) results[[length(results) + 1]] <- r
}

# ============================================
# SURVIVAL: AFT Model
# ============================================
message("\n=== Survival: AFT ===")

if (requireNamespace("survival", quietly = TRUE)) {
  suppressPackageStartupMessages(library(survival))

  for (n in c(100, 500, 1000)) {
    d <- generate_survival_data(n)
    local_d <- d

    r <- run_unified("AFT", "Weibull", n, 42, "survival", function() {
      survreg(Surv(time, event) ~ x1 + x2, data = local_d, dist = "weibull")
    }, function(fit) {
      list(
        coefficients = as.numeric(coef(fit)),
        log_likelihood = as.numeric(logLik(fit))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: survival package not installed")
}

# ============================================
# SURVIVAL: Competing Risks
# ============================================
message("\n=== Survival: Competing Risks ===")

if (requireNamespace("cmprsk", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    set.seed(42)
    time_vals <- runif(n, 0.1, 20)
    u <- runif(n)
    event_type <- ifelse(u < 0.3, 0, ifelse(u < 0.65, 1, 2))

    local_time <- time_vals
    local_event <- event_type

    r <- run_unified("Competing_Risks", "Aalen-Johansen", n, 42, "competing_risks", function() {
      cmprsk::cuminc(local_time, local_event)
    }, function(fit) {
      list(
        n_obs = n,
        n_event_types = length(unique(local_event[local_event > 0]))
      )
    }, iterations = 20)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: cmprsk package not installed")
}

# ============================================
# CAUSAL: RD Multi-cutoff
# ============================================
message("\n=== RD Multi-cutoff ===")

if (requireNamespace("rdrobust", quietly = TRUE)) {
  for (n in c(500, 1000)) {
    set.seed(42)
    cutoffs <- c(0.0, 2.0)
    half <- n %/% 2
    x_vals <- numeric(n)
    y_vals <- numeric(n)
    c_idx <- integer(n)
    for (i in 1:n) {
      ci <- if (i <= half) 1 else 2
      c_val <- cutoffs[ci]
      x <- c_val + runif(1, -2, 2)
      te <- if (x >= c_val) 1.5 + (ci - 1) else 0.0
      y <- 0.5 + 0.3 * x + te + runif(1, -0.5, 0.5)
      x_vals[i] <- x
      y_vals[i] <- y
      c_idx[i] <- ci
    }

    local_x <- x_vals
    local_y <- y_vals
    local_c <- c_idx

    r <- run_unified("RD_Multi", "2-cutoff", n, 42, "rd_multi", function() {
      # Run separate rdrobust at each cutoff
      coefs <- numeric(2)
      for (j in 1:2) {
        mask <- local_c == j
        fit <- rdrobust::rdrobust(local_y[mask], local_x[mask], c = cutoffs[j])
        coefs[j] <- fit$coef[1]
      }
      coefs
    }, function(result) {
      list(coefficients = as.numeric(result))
    }, iterations = 10)
    if (!is.null(r)) results[[length(results) + 1]] <- r
  }
} else {
  message("  Skipping: rdrobust package not installed")
}

# ============================================
# Save all results to JSON
# ============================================
message("\n=== Saving results ===")

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_file <- sprintf("results/unified_r_%s.json", timestamp)

# Clean up NULL entries
results <- Filter(Negate(is.null), results)

# Sanitize outputs: convert any non-serializable types (table, matrix, etc.) to plain vectors
sanitize <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) return(lapply(x, sanitize))
  if (is.table(x) || is.matrix(x)) return(as.numeric(x))
  if (is.factor(x)) return(as.character(x))
  x
}
results <- lapply(results, sanitize)

json_out <- toJSON(results, auto_unbox = TRUE, pretty = TRUE, digits = 6, force = TRUE)
writeLines(json_out, output_file)

message(sprintf("\nResults saved to: %s", output_file))
message(sprintf("Total benchmarks: %d", length(results)))
message(sprintf("Data CSVs saved to: %s", data_dir))
message(sprintf("R version: %s", R.version.string))

# Print summary table
message("\n=== Summary ===")
for (r in results) {
  message(sprintf("  %-25s %-15s n=%-6d median=%.1f us  mem=%s",
                  r$method, r$variant, r$n,
                  r$time_median_us,
                  if (is.na(r$mem_alloc_bytes)) "NA"
                  else if (r$mem_alloc_bytes < 1024) sprintf("%d B", r$mem_alloc_bytes)
                  else if (r$mem_alloc_bytes < 1024^2) sprintf("%.1f KB", r$mem_alloc_bytes / 1024)
                  else sprintf("%.2f MB", r$mem_alloc_bytes / 1024^2)))
}
