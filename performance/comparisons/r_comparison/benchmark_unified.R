#!/usr/bin/env Rscript
# Unified R Benchmarks — mirrors crates/p2a-core/benches/unified_benchmarks.rs
#
# Uses the SAME DGPs, sample sizes, and method configurations as the Rust
# unified benchmark.  Output: a single CSV with distribution statistics
# (min, p25, median, p75, max, mean, std) plus memory, ready to merge.
#
# Run:   Rscript benchmark_unified.R
#
# Required packages (install once):
#   install.packages(c("bench", "sandwich", "plm", "lfe", "fixest",
#       "forecast", "nlme", "quantreg", "boot", "MASS", "nnet",
#       "survival", "AER", "ivreg", "rdrobust", "MatchIt",
#       "vars", "tseries", "rugarch", "dbscan", "cluster",
#       "Rtsne", "randomForest", "e1071", "lmtest", "car",
#       "spdep", "spatialreg"))

# ============================================
# Setup
# ============================================

suppressPackageStartupMessages(library(bench))
set.seed(42)

cat("=== R Unified Benchmarks ===\n\n")

results <- list()
idx <- 1

# ============================================
# Helpers
# ============================================

run_bench <- function(method, variant, n, fn, iterations = 100, slow = FALSE) {
  if (slow) iterations <- min(iterations, 20)
  res <- tryCatch({
    bm <- bench::mark(fn(), iterations = iterations, check = FALSE,
                       memory = TRUE, filter_gc = FALSE)
    raw_times <- as.numeric(bm$time[[1]]) * 1e6
    mem <- as.numeric(bm$mem_alloc[[1]])
    list(
      method = method, variant = variant, n = n,
      iterations = length(raw_times),
      time_min_us = min(raw_times),
      time_p25_us = unname(quantile(raw_times, 0.25)),
      time_median_us = median(raw_times),
      time_p75_us = unname(quantile(raw_times, 0.75)),
      time_max_us = max(raw_times),
      time_mean_us = mean(raw_times),
      time_std_us = sd(raw_times),
      itr_per_sec = 1e6 / median(raw_times),
      mem_alloc_bytes = mem
    )
  }, error = function(e) {
    cat(sprintf("  ERROR [%s/%s n=%d]: %s\n", method, variant, n, e$message))
    NULL
  })
  if (!is.null(res)) {
    cat(sprintf("  %-30s n=%6d  median: %10.1f us\n", paste0(method, "/", variant), n, res$time_median_us))
  }
  res
}

add_result <- function(r) {
  if (!is.null(r)) {
    results[[idx]] <<- r
    idx <<- idx + 1
  }
}

# ============================================
# Data Generators (matching Rust DGP, seed=42)
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
  ts(trend + seasonal + noise, frequency = 12)
}

generate_cluster_data <- function(n, k = 5) {
  set.seed(42)
  data <- matrix(0, nrow = n, ncol = k)
  for (i in 1:n) {
    cluster <- (i - 1) %% 3
    center <- cluster * 3
    data[i, ] <- center + runif(k, -0.5, 0.5)
  }
  data
}

generate_did_data <- function(n) {
  set.seed(42)
  half <- n / 2
  treatment <- c(rep(0, half), rep(1, half))
  post <- rep(c(0, 1), n / 2)
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
  prob <- 1 / (1 + exp(-0.3 * x1 - 0.2 * x2))
  treatment <- as.numeric(runif(n) < prob)
  y <- 1.0 + 0.5 * treatment + 0.3 * x1 + 0.2 * x2 + runif(n, -0.5, 0.5)
  data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2)
}

generate_count_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  group <- (0:(n-1)) %% 5
  lambda <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
  y <- rpois(n, lambda)
  data.frame(y = y, x1 = x1, x2 = x2, group = factor(group))
}

generate_zeroinfl_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  lambda <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
  structural_zero <- runif(n) < 0.3
  y <- ifelse(structural_zero, 0L, rpois(n, lambda))
  data.frame(y = y, x1 = x1, x2 = x2)
}

generate_ordered_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  u <- runif(n, 0.001, 0.999)
  logistic_noise <- log(u / (1 - u))
  y_star <- 0.5 * x1 + 0.3 * x2 + logistic_noise
  y <- factor(ifelse(y_star < -0.5, "1", ifelse(y_star < 0.5, "2", "3")), ordered = TRUE)
  data.frame(y = y, x1 = x1, x2 = x2)
}

generate_multinomial_data <- function(n) {
  set.seed(42)
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  u1 <- 0.5 * x1 + 0.2 * x2 + runif(n, -1, 1)
  u2 <- -0.3 * x1 + 0.4 * x2 + runif(n, -1, 1)
  u3 <- runif(n, -1, 1)
  y <- ifelse(u1 >= u2 & u1 >= u3, "A", ifelse(u2 >= u3, "B", "C"))
  data.frame(y = factor(y), x1 = x1, x2 = x2)
}

generate_garch_data <- function(n) {
  set.seed(42)
  omega <- 0.1; alpha <- 0.15; beta <- 0.75
  y <- numeric(n)
  sigma2 <- numeric(n)
  sigma2[1] <- omega / (1 - alpha - beta)
  y[1] <- sqrt(sigma2[1]) * rnorm(1)
  for (t in 2:n) {
    sigma2[t] <- omega + alpha * y[t-1]^2 + beta * sigma2[t-1]
    y[t] <- sqrt(sigma2[t]) * rnorm(1)
  }
  y
}

generate_bivariate_data <- function(n) {
  set.seed(42)
  y1 <- numeric(n); y2 <- numeric(n)
  y1[1] <- rnorm(1); y2[1] <- rnorm(1)
  for (t in 2:n) {
    y1[t] <- 0.5 * y1[t-1] + 0.1 * y2[t-1] + rnorm(1, sd = 0.5)
    y2[t] <- 0.2 * y1[t-1] + 0.6 * y2[t-1] + rnorm(1, sd = 0.5)
  }
  data.frame(y1 = y1, y2 = y2)
}

generate_survival_data <- function(n) {
  set.seed(42)
  time_vals <- runif(n, 0.1, 20.0)
  event <- rbinom(n, 1, 0.7)
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  group <- ifelse((0:(n-1)) %% 2 == 0, "A", "B")
  data.frame(time = time_vals, event = event, x1 = x1, x2 = x2, group = group)
}

generate_staggered_panel <- function(n_units, n_periods) {
  set.seed(42)
  n <- n_units * n_periods
  unit <- rep(0:(n_units - 1), each = n_periods)
  time <- rep(0:(n_periods - 1), times = n_units)
  treat_time <- integer(n)
  treated <- numeric(n)
  y <- numeric(n)
  for (u in 0:(n_units - 1)) {
    tt <- if (u < n_units / 3) 0L else as.integer((n_periods / 3) + (u %% (n_periods / 2)) + 1)
    unit_effect <- u * 0.1
    for (t_idx in 0:(n_periods - 1)) {
      i <- u * n_periods + t_idx + 1
      treat_time[i] <- tt
      is_treated <- tt > 0 & t_idx >= tt
      treated[i] <- ifelse(is_treated, 1, 0)
      te <- ifelse(is_treated, 2.0, 0.0)
      y[i] <- unit_effect + 0.05 * t_idx + te + runif(1, -0.5, 0.5)
    }
  }
  data.frame(unit = unit, time = time, y = y, treat_time = treat_time, treated = treated)
}


# ============================================
# REGRESSION [n = 100, 1000, 10000]
# ============================================
cat("\n--- Regression ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_regression_data(n)

  # OLS Standard
  add_result(run_bench("OLS", "standard", n, function() lm(y ~ x1+x2+x3+x4+x5, data = data)))

  # OLS HC1
  if (requireNamespace("sandwich", quietly = TRUE)) {
    add_result(run_bench("OLS", "HC1", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      sandwich::vcovHC(fit, type = "HC1")
    }))
  }
}

# ============================================
# Regression Diagnostics & Variants [n = 100, 1000, 10000]
# ============================================
cat("\n--- Regression Diagnostics & Variants ---\n")

for (n in c(100, 1000, 10000)) {
  data <- generate_regression_data(n)

  # OLS_HC0, HC2, HC3
  if (requireNamespace("sandwich", quietly = TRUE)) {
    for (hc_type in c("HC0", "HC2", "HC3")) {
      add_result(run_bench(paste0("OLS_", hc_type), "robust", n, function() {
        fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
        sandwich::vcovHC(fit, type = hc_type)
      }))
    }
  }

  # OLS_HAC (Newey-West)
  if (requireNamespace("sandwich", quietly = TRUE)) {
    add_result(run_bench("OLS_HAC", "Newey-West", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      sandwich::NeweyWest(fit)
    }))
  }

  # Breusch-Godfrey
  if (requireNamespace("lmtest", quietly = TRUE)) {
    add_result(run_bench("Breusch_Godfrey", "LM", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      lmtest::bgtest(fit, order = 1)
    }))
  }

  # Breusch-Pagan
  if (requireNamespace("lmtest", quietly = TRUE)) {
    add_result(run_bench("Breusch_Pagan", "test", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      lmtest::bptest(fit)
    }))
  }

  # Durbin-Watson
  if (requireNamespace("lmtest", quietly = TRUE)) {
    add_result(run_bench("Durbin_Watson", "test", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      lmtest::dwtest(fit)
    }))
  }

  # RESET
  if (requireNamespace("lmtest", quietly = TRUE)) {
    add_result(run_bench("RESET", "test", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      lmtest::resettest(fit, power = 2:3)
    }))
  }

  # VIF
  if (requireNamespace("car", quietly = TRUE)) {
    add_result(run_bench("VIF", "diagnostics", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      car::vif(fit)
    }))
  }

  # Jarque-Bera
  if (requireNamespace("tseries", quietly = TRUE)) {
    add_result(run_bench("Jarque_Bera", "standalone", n, function() {
      fit <- lm(y ~ x1+x2+x3+x4+x5, data = data)
      tseries::jarque.bera.test(residuals(fit))
    }))
  }
}

# OLS_Bootstrap [n = 100, 1000] (slow)
if (requireNamespace("boot", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    data <- generate_regression_data(n)
    add_result(run_bench("OLS_Bootstrap", "pairs", n, function() {
      boot::boot(data, function(d, i) {
        coef(lm(y ~ x1+x2+x3+x4+x5, data = d[i, ]))
      }, R = 199)
    }, slow = TRUE))
  }
}

# GLS AR1 [n = 100, 1000] (slow, O(n^3))
if (requireNamespace("nlme", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    data <- generate_regression_data(n, k = 1)
    add_result(run_bench("GLS", "AR1", n, function() {
      nlme::gls(y ~ x1, data = data, correlation = nlme::corAR1(0.5, form = ~ 1))
    }, slow = TRUE))
  }
}

# Quantile Regression [n = 100, 1000] (slow)
if (requireNamespace("quantreg", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    data <- generate_regression_data(n, k = 3)
    add_result(run_bench("Quantile_Regression", "median", n, function() {
      quantreg::rq(y ~ x1+x2+x3, data = data, tau = 0.5)
    }, slow = TRUE))
  }
}

# Smooth Spline [n = 100, 1000]
for (n in c(100, 1000)) {
  data <- generate_regression_data(n, k = 1)
  add_result(run_bench("Smooth_Spline", "GCV", n, function() {
    smooth.spline(data$x1, data$y)
  }))
}

# Stepwise [n = 100, 1000] (slow)
for (n in c(100, 1000)) {
  data <- generate_regression_data(n)
  add_result(run_bench("Stepwise", "both_AIC", n, function() {
    full <- lm(y ~ x1+x2+x3+x4+x5, data = data)
    step(full, direction = "both", trace = 0)
  }, slow = TRUE))
}

# NLS [n = 100, 500, 1000] (slow)
for (n in c(100, 500, 1000)) {
  set.seed(42)
  x_vals <- (0:(n-1)) * 0.1
  y_vals <- 2.0 * exp(0.3 * x_vals) + rnorm(n, sd = 0.1)
  nls_data <- data.frame(y = y_vals, x = x_vals)
  add_result(run_bench("NLS", "exp_growth", n, function() {
    nls(y ~ a * exp(b * x), data = nls_data, start = list(a = 1, b = 0.1))
  }, slow = TRUE))
}

# LOESS [n = 100, 1000, 10000]
cat("\n--- LOESS ---\n")
for (n in c(100, 1000, 10000)) {
  data <- generate_regression_data(n, k = 1)
  add_result(run_bench("LOESS", "span=0.75", n, function() {
    loess(y ~ x1, data = data, span = 0.75)
  }))
}

# OLS_Clustered [n = 100, 500, 1000]
if (requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    entities <- n / 10
    nn <- entities * 10
    set.seed(42)
    entity <- rep(0:(entities - 1), each = 10)
    x1 <- rnorm(nn); x2 <- rnorm(nn)
    y <- entity * 0.1 + 0.5 * x1 + 0.3 * x2 + rnorm(nn, sd = 0.5)
    cldata <- data.frame(entity = entity, y = y, x1 = x1, x2 = x2)
    add_result(run_bench("OLS_Clustered", "entity", nn, function() {
      fit <- lm(y ~ x1 + x2, data = cldata)
      sandwich::vcovCL(fit, cluster = cldata$entity)
    }))
  }
}

# OLS_Driscoll_Kraay [n = 100, 500, 1000]
if (requireNamespace("plm", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    entities <- n / 10
    nn <- entities * 10
    set.seed(42)
    entity <- rep(0:(entities - 1), each = 10)
    time_col <- rep(0:9, times = entities)
    x1 <- rnorm(nn); x2 <- rnorm(nn)
    y <- entity * 0.1 + 0.5 * x1 + 0.3 * x2 + rnorm(nn, sd = 0.5)
    dkdata <- data.frame(entity = factor(entity), time = factor(time_col), y = y, x1 = x1, x2 = x2)
    pdata <- plm::pdata.frame(dkdata, index = c("entity", "time"))
    add_result(run_bench("OLS_Driscoll_Kraay", "Bartlett", nn, function() {
      fit <- plm::plm(y ~ x1 + x2, data = pdata, model = "pooling")
      plm::vcovSCC(fit)
    }))
  }
}

# Marginal Effects [n = 100, 500, 1000]
if (requireNamespace("margins", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    set.seed(42)
    x1 <- rnorm(n); x2 <- rnorm(n)
    logit_val <- -1 + 0.5 * x1 + 0.3 * x2
    prob <- 1 / (1 + exp(-logit_val))
    y <- rbinom(n, 1, prob)
    me_data <- data.frame(y = y, x1 = x1, x2 = x2)
    add_result(run_bench("Marginal_Effects", "logit_AME", n, function() {
      fit <- glm(y ~ x1 + x2, data = me_data, family = binomial(link = "logit"))
      margins::margins(fit)
    }, slow = TRUE))
  }
}


# ============================================
# PANEL DATA
# ============================================
cat("\n--- Panel Data ---\n")

if (requireNamespace("plm", quietly = TRUE)) {
  for (params in list(c(10, 10), c(50, 20), c(100, 100))) {
    n_ent <- params[1]; n_per <- params[2]; n <- n_ent * n_per
    data <- generate_panel_data(n_ent, n_per)
    pdata <- plm::pdata.frame(data, index = c("entity", "time"))

    # Fixed Effects
    add_result(run_bench("FixedEffects", "within", n, function() plm::plm(y ~ x1 + x2, data = pdata, model = "within")))
    # Random Effects
    add_result(run_bench("RandomEffects", "GLS", n, function() plm::plm(y ~ x1 + x2, data = pdata, model = "random")))
    # Hausman test
    add_result(run_bench("Hausman", "phtest", n, function() {
      fe <- plm::plm(y ~ x1 + x2, data = pdata, model = "within")
      re <- plm::plm(y ~ x1 + x2, data = pdata, model = "random")
      plm::phtest(fe, re)
    }, slow = TRUE))
  }

  # HDFE (lfe or fixest)
  if (requireNamespace("lfe", quietly = TRUE)) {
    for (params in list(c(10, 10), c(50, 20), c(100, 100))) {
      n_ent <- params[1]; n_per <- params[2]; n <- n_ent * n_per
      data <- generate_panel_data(n_ent, n_per)
      add_result(run_bench("HDFE", "2-way", n, function() lfe::felm(y ~ x1 + x2 | entity + time, data = data)))
    }
  }

  # Panel GLS [n = 100, 1000] (no n=10000)
  for (params in list(c(10, 10), c(50, 20))) {
    n_ent <- params[1]; n_per <- params[2]; n <- n_ent * n_per
    data <- generate_panel_data(n_ent, n_per)
    pdata <- plm::pdata.frame(data, index = c("entity", "time"))
    add_result(run_bench("Panel_GLS", "pggls", n, function() plm::pggls(y ~ x1 + x2, data = pdata)))
  }

  # Arellano-Bond [n = 100, 1000] (slow, no n=10000)
  for (params in list(c(10, 10), c(50, 20))) {
    n_ent <- params[1]; n_per <- params[2]; n <- n_ent * n_per
    data <- generate_panel_data(n_ent, n_per)
    pdata <- plm::pdata.frame(data, index = c("entity", "time"))
    add_result(run_bench("Arellano_Bond", "pgmm", n, function() {
      plm::pgmm(y ~ lag(y, 1) + x1 + x2 | lag(y, 2:99), data = pdata,
                 effect = "individual", model = "twosteps")
    }, slow = TRUE))
  }
}

# FEGLM_Gaussian [n = 100, 500, 1000]
if (requireNamespace("fixest", quietly = TRUE)) {
  for (n in c(100, 500, 1000)) {
    entities <- n / 10; nn <- entities * 10
    set.seed(42)
    entity <- factor(rep(paste0("e", 0:(entities-1)), each = 10))
    x1 <- rnorm(nn); x2 <- rnorm(nn)
    y <- as.numeric(entity) * 0.1 + 0.5 * x1 + 0.3 * x2 + rnorm(nn, sd = 0.5)
    fedata <- data.frame(entity = entity, y = y, x1 = x1, x2 = x2)
    add_result(run_bench("FEGLM_Gaussian", "entity_FE", nn, function() {
      fixest::feglm(y ~ x1 + x2 | entity, data = fedata, family = gaussian())
    }))
  }
}

# ANOVA_TwoWay [n = 100, 500, 1000]
for (n in c(100, 500, 1000)) {
  set.seed(42)
  factorA <- factor(paste0("A", (0:(n-1)) %% 3))
  factorB <- factor(paste0("B", (0:(n-1)) %% 2))
  y <- ((0:(n-1)) %% 3) * 0.5 + ((0:(n-1)) %% 2) * 0.3 + rnorm(n)
  anova_data <- data.frame(y = y, factorA = factorA, factorB = factorB)
  add_result(run_bench("ANOVA_TwoWay", "with_interaction", n, function() {
    summary(aov(y ~ factorA * factorB, data = anova_data))
  }))
}


# ============================================
# DISCRETE CHOICE [n = 100, 1000, 10000]
# ============================================
cat("\n--- Discrete Choice ---\n")

for (n in c(100, 1000, 10000)) {
  bdata <- generate_binary_data(n)
  add_result(run_bench("Logit", "MLE", n, function() glm(y ~ x1+x2, data = bdata, family = binomial("logit"))))
  add_result(run_bench("Probit", "MLE", n, function() glm(y ~ x1+x2, data = bdata, family = binomial("probit"))))
}

# Count models [n = 100, 1000, 10000]
cat("\n--- Count & Zero-Inflated Models ---\n")
for (n in c(100, 1000, 10000)) {
  cdata <- generate_count_data(n)
  zidata <- generate_zeroinfl_data(n)

  # Poisson
  add_result(run_bench("Poisson", "GLM", n, function() glm(y ~ x1+x2, data = cdata, family = poisson()), slow = TRUE))

  # NegBin
  if (requireNamespace("MASS", quietly = TRUE)) {
    add_result(run_bench("NegBin", "MLE", n, function() MASS::glm.nb(y ~ x1+x2, data = cdata), slow = TRUE))
  }

  # ZIP
  if (requireNamespace("pscl", quietly = TRUE)) {
    add_result(run_bench("ZIP", "EM", n, function() pscl::zeroinfl(y ~ x1+x2 | 1, data = zidata, dist = "poisson"), slow = TRUE))
  }

  # ZINB
  if (requireNamespace("pscl", quietly = TRUE)) {
    add_result(run_bench("ZINB", "EM", n, function() pscl::zeroinfl(y ~ x1+x2 | 1, data = zidata, dist = "negbin"), slow = TRUE))
  }

  # Hurdle (Poisson)
  if (requireNamespace("pscl", quietly = TRUE)) {
    add_result(run_bench("Hurdle", "Poisson", n, function() pscl::hurdle(y ~ x1+x2 | x1+x2, data = zidata, dist = "poisson"), slow = TRUE))
  }
}

# Ordered & Multinomial [n = 100, 1000, 10000]
cat("\n--- Ordered & Multinomial ---\n")
for (n in c(100, 1000, 10000)) {
  odata <- generate_ordered_data(n)
  mdata <- generate_multinomial_data(n)

  if (requireNamespace("MASS", quietly = TRUE)) {
    add_result(run_bench("Ordered_Logit", "MLE", n, function() MASS::polr(y ~ x1+x2, data = odata), slow = TRUE))
  }
  if (requireNamespace("nnet", quietly = TRUE)) {
    add_result(run_bench("Multinomial_Logit", "MLE", n, function() {
      suppressMessages(nnet::multinom(y ~ x1+x2, data = mdata, trace = FALSE))
    }, slow = TRUE))
  }
}


# ============================================
# TIME SERIES [n = 100, 1000, 10000]
# ============================================
cat("\n--- Time Series ---\n")

for (n in c(100, 1000, 10000)) {
  ts_data <- generate_time_series(n)

  # ARIMA
  if (requireNamespace("forecast", quietly = TRUE)) {
    add_result(run_bench("ARIMA", "(1,1,1)", n, function() forecast::Arima(ts_data, order = c(1, 1, 1)), slow = TRUE))
  }

  # MSTL
  if (requireNamespace("forecast", quietly = TRUE)) {
    add_result(run_bench("MSTL", "period=12", n, function() forecast::mstl(ts_data)))
  }

  # STL
  if (n >= 24) {
    add_result(run_bench("STL", "period=12", n, function() stl(ts_data, s.window = "periodic")))
  }

  # Decompose
  if (n >= 24) {
    add_result(run_bench("Decompose", "additive", n, function() decompose(ts_data, type = "additive")))
  }

  # Holt-Winters
  ts_pos <- ts_data + 5.0  # ensure positive
  add_result(run_bench("Holt_Winters", "additive", n, function() HoltWinters(ts_pos, seasonal = "additive"), slow = TRUE))

  # ACF
  add_result(run_bench("ACF", "correlation", n, function() acf(as.numeric(ts_data), lag.max = 20, plot = FALSE)))

  # PACF
  add_result(run_bench("PACF", "durbin_levinson", n, function() pacf(as.numeric(ts_data), lag.max = 20, plot = FALSE)))

  # Phillips-Perron
  if (requireNamespace("tseries", quietly = TRUE)) {
    add_result(run_bench("Phillips_Perron", "short_lag", n, function() tseries::pp.test(as.numeric(ts_data))))
  }

  # Box-Ljung
  add_result(run_bench("Box_Ljung", "lag=10", n, function() Box.test(as.numeric(ts_data), lag = 10, type = "Ljung-Box")))

  # AR
  add_result(run_bench("AR", "yule_walker", n, function() ar(as.numeric(ts_data), aic = TRUE, order.max = 5, method = "yule-walker")))
}

# GARCH [n = 100, 1000] (slow)
if (requireNamespace("rugarch", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    garch_y <- generate_garch_data(n)
    spec <- rugarch::ugarchspec(variance.model = list(garchOrder = c(1, 1)),
                                mean.model = list(armaOrder = c(0, 0), include.mean = FALSE))
    add_result(run_bench("GARCH", "(1,1)", n, function() {
      rugarch::ugarchfit(spec, garch_y, solver = "hybrid")
    }, slow = TRUE))
  }
}

# Kalman [n = 100, 1000]
if (requireNamespace("dlm", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ts_data <- generate_time_series(n)
    y_vec <- as.numeric(ts_data)
    add_result(run_bench("Kalman", "local_level", n, function() {
      mod <- dlm::dlmModPoly(order = 1, dV = 0.5, dW = 0.1, m0 = y_vec[1], C0 = 1)
      dlm::dlmFilter(y_vec, mod)
    }))
  }
} else if (requireNamespace("KFAS", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ts_data <- generate_time_series(n)
    y_vec <- as.numeric(ts_data)
    add_result(run_bench("Kalman", "local_level", n, function() {
      mod <- KFAS::SSModel(y_vec ~ KFAS::SSMtrend(1, Q = 0.1), H = 0.5)
      KFAS::KFS(mod)
    }))
  }
}

# StructTS [n = 100, 1000] (slow)
for (n in c(100, 1000)) {
  ts_data <- generate_time_series(n)
  add_result(run_bench("StructTS", "trend", n, function() StructTS(ts_data, type = "trend"), slow = TRUE))
}

# VAR [n = 100, 1000]
if (requireNamespace("vars", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    bdata <- generate_bivariate_data(n)
    add_result(run_bench("VAR", "p=1", n, function() vars::VAR(bdata, p = 1), slow = TRUE))
  }
}

# VECM [n = 100, 1000]
if (requireNamespace("vars", quietly = TRUE) && requireNamespace("urca", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    bdata <- generate_bivariate_data(n)
    add_result(run_bench("VECM", "rank=1", n, function() {
      jt <- urca::ca.jo(bdata, type = "eigen", K = 2, ecdet = "const")
      urca::cajorls(jt, r = 1)
    }, slow = TRUE))
  }
}

# Granger [n = 100, 1000]
if (requireNamespace("lmtest", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    bdata <- generate_bivariate_data(n)
    add_result(run_bench("Granger", "lags=4", n, function() lmtest::grangertest(y1 ~ y2, data = bdata, order = 4)))
  }
}

# CCF [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  y1 <- 0.01 * (0:(n-1)) + sin((0:(n-1)) * pi / 6) * 2 + runif(n, 0, 0.5)
  y2 <- 0.02 * (0:(n-1)) + cos((0:(n-1)) * pi / 4) * 1.5 + runif(n, 0, 0.5)
  add_result(run_bench("CCF", "cross_correlation", n, function() ccf(y1, y2, lag.max = 10, plot = FALSE)))
}

# Changepoint [n = 100, 1000, 10000]
if (requireNamespace("changepoint", quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    ts_data <- generate_time_series(n)
    add_result(run_bench("Changepoint", "PELT", n, function() changepoint::cpt.mean(as.numeric(ts_data), method = "PELT")))
  }
}


# ============================================
# MACHINE LEARNING [n = 100, 1000, 10000]
# ============================================
cat("\n--- Machine Learning ---\n")

for (n in c(100, 1000, 10000)) {
  ml_data <- generate_cluster_data(n)

  add_result(run_bench("K-Means", "k=3", n, function() kmeans(ml_data, centers = 3, nstart = 5, iter.max = 100)))
  add_result(run_bench("PCA", "k=3", n, function() prcomp(ml_data, center = TRUE, scale. = FALSE)))
}

# Hierarchical [n = 100, 1000]
for (n in c(100, 1000)) {
  ml_data <- generate_cluster_data(n)
  add_result(run_bench("Hierarchical", "Ward", n, function() {
    d <- dist(ml_data)
    hclust(d, method = "ward.D2")
  }))
}

# DBSCAN [n = 100, 1000]
if (requireNamespace("dbscan", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ml_data <- generate_cluster_data(n)
    add_result(run_bench("DBSCAN", "eps=1.5", n, function() dbscan::dbscan(ml_data, eps = 1.5, minPts = 5)))
  }
}

# Random Forest [n = 100, 1000]
if (requireNamespace("randomForest", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ml_data <- generate_cluster_data(n)
    target <- ml_data[, 1]
    features <- ml_data[, 2:ncol(ml_data)]
    add_result(run_bench("RandomForest", "100trees", n, function() {
      randomForest::randomForest(features, target, ntree = 100, maxnodes = 10, nodesize = 5)
    }))
  }
}

# K_Medoids (PAM) [n = 100, 1000]
if (requireNamespace("cluster", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ml_data <- generate_cluster_data(n)
    add_result(run_bench("K_Medoids", "PAM", n, function() cluster::pam(ml_data, k = 3)))
  }
}

# SVM [n = 100, 1000, 10000]
if (requireNamespace("e1071", quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    ml_data <- generate_cluster_data(n)
    labels <- factor(ifelse((0:(n-1)) %% 3 == 0, 1, -1))
    add_result(run_bench("SVM", "linear", n, function() e1071::svm(ml_data, labels, kernel = "linear", cost = 1)))
  }
}

# t-SNE [n = 100, 1000] (slow)
if (requireNamespace("Rtsne", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ml_data <- generate_cluster_data(n)
    perp <- min(30, n / 3 - 1)
    add_result(run_bench("tSNE", "default", n, function() {
      Rtsne::Rtsne(ml_data, dims = 2, perplexity = perp, max_iter = 500, eta = 200, check_duplicates = FALSE)
    }, slow = TRUE))
  }
}

# Silhouette [n = 100, 1000]
if (requireNamespace("cluster", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ml_data <- generate_cluster_data(n)
    km <- kmeans(ml_data, centers = 3, nstart = 1)
    add_result(run_bench("Silhouette", "from_kmeans", n, function() cluster::silhouette(km$cluster, dist(ml_data))))
  }
}

# MDS [n = 100, 1000]
for (n in c(100, 1000)) {
  ml_data <- generate_cluster_data(n)
  add_result(run_bench("MDS", "classical", n, function() cmdscale(dist(ml_data), k = 2)))
}

# Factor Analysis [n = 100, 1000]
for (n in c(100, 1000)) {
  set.seed(42)
  p <- 10; k <- 3
  fadata <- matrix(0, nrow = n, ncol = p)
  f1 <- runif(n, -2, 2); f2 <- runif(n, -2, 2); f3 <- runif(n, -2, 2)
  fadata[, 1] <- 0.8 * f1 + runif(n, -0.3, 0.3)
  fadata[, 2] <- 0.7 * f1 + runif(n, -0.4, 0.4)
  fadata[, 3] <- 0.75 * f1 + runif(n, -0.35, 0.35)
  fadata[, 4] <- 0.8 * f2 + runif(n, -0.3, 0.3)
  fadata[, 5] <- 0.7 * f2 + runif(n, -0.4, 0.4)
  fadata[, 6] <- 0.75 * f2 + runif(n, -0.35, 0.35)
  fadata[, 7] <- 0.8 * f3 + runif(n, -0.3, 0.3)
  fadata[, 8] <- 0.7 * f3 + runif(n, -0.4, 0.4)
  fadata[, 9] <- 0.75 * f3 + runif(n, -0.35, 0.35)
  fadata[, 10] <- runif(n, -1, 1)
  add_result(run_bench("factanal", "varimax", n, function() factanal(fadata, factors = k, rotation = "varimax")))
}


# ============================================
# CAUSAL INFERENCE [mostly n = 100, 1000]
# ============================================
cat("\n--- Causal Inference ---\n")

# DiD [n = 100, 1000]
for (n in c(100, 1000)) {
  ddata <- generate_did_data(n)
  add_result(run_bench("DiD", "canonical", n, function() {
    lm(y ~ treatment * post + x1, data = ddata)
  }, slow = TRUE))
}

# IV/2SLS [n = 100, 1000]
if (requireNamespace("ivreg", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ivdata <- generate_iv_data(n)
    add_result(run_bench("IV_2SLS", "2sls", n, function() {
      ivreg::ivreg(y ~ x_endog + x_exog | instrument + x_exog, data = ivdata)
    }, slow = TRUE))
  }
} else if (requireNamespace("AER", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    ivdata <- generate_iv_data(n)
    add_result(run_bench("IV_2SLS", "2sls", n, function() {
      AER::ivreg(y ~ x_endog + x_exog | instrument + x_exog, data = ivdata)
    }, slow = TRUE))
  }
}

# RD [n = 100, 1000]
if (requireNamespace("rdrobust", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    rddata <- generate_rd_data(n)
    add_result(run_bench("RD", "sharp", n, function() rdrobust::rdrobust(rddata$y, rddata$running, c = 0), slow = TRUE))
  }
}

# Matching [n = 100, 1000]
if (requireNamespace("MatchIt", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    tdata <- generate_treatment_data(n)
    add_result(run_bench("Matching", "nearest", n, function() {
      MatchIt::matchit(treatment ~ x1 + x2, data = tdata, method = "nearest")
    }, slow = TRUE))
  }
}

# IPW [n = 100, 1000]
for (n in c(100, 1000)) {
  tdata <- generate_treatment_data(n)
  add_result(run_bench("IPW", "ATE", n, function() {
    ps <- glm(treatment ~ x1 + x2, data = tdata, family = binomial)$fitted.values
    w1 <- tdata$treatment / ps
    w0 <- (1 - tdata$treatment) / (1 - ps)
    ate <- weighted.mean(tdata$y, w1) - weighted.mean(tdata$y, w0)
    ate
  }, slow = TRUE))
}

# CBPS [n = 100, 1000]
if (requireNamespace("CBPS", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    tdata <- generate_treatment_data(n)
    add_result(run_bench("CBPS", "exact", n, function() {
      CBPS::CBPS(treatment ~ x1 + x2, data = tdata)
    }, slow = TRUE))
  }
}

# WeightIt [n = 100, 1000]
if (requireNamespace("WeightIt", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    tdata <- generate_treatment_data(n)
    add_result(run_bench("WeightIt", "logistic", n, function() {
      WeightIt::weightit(treatment ~ x1 + x2, data = tdata, method = "ps")
    }, slow = TRUE))
  }
}

# Sensemakr [n = 100, 1000, 10000]
if (requireNamespace("sensemakr", quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    tdata <- generate_treatment_data(n)
    add_result(run_bench("Sensemakr", "sensitivity", n, function() {
      fit <- lm(y ~ treatment + x1 + x2, data = tdata)
      sensemakr::sensemakr(fit, treatment = "treatment", benchmark_covariates = "x1", kd = 1)
    }))
  }
}

# Synthetic Control [n = 100, 1000]
if (requireNamespace("Synth", quietly = TRUE)) {
  for (params in list(c(10, 10), c(50, 20))) {
    n_units <- params[1]; n_periods <- params[2]; n <- n_units * n_periods
    set.seed(42)
    unit_ids <- rep(1:n_units, each = n_periods)
    time_ids <- rep(1:n_periods, times = n_units)
    outcome <- numeric(n)
    pred1 <- runif(n, 0, 1); pred2 <- runif(n, 0, 1)
    for (u in 1:n_units) {
      for (t_idx in 1:n_periods) {
        i <- (u - 1) * n_periods + t_idx
        base <- (u - 1) * 0.5 + (t_idx - 1) * 0.1
        te <- ifelse(u == 1 & t_idx >= 8, 2.0, 0)
        outcome[i] <- base + te + runif(1, -0.3, 0.3)
      }
    }
    synth_data <- data.frame(unit = unit_ids, time = time_ids, outcome = outcome, pred1 = pred1, pred2 = pred2)
    add_result(run_bench("SynthControl", "Nelder-Mead", n, function() {
      dp <- Synth::dataprep(foo = synth_data, predictors = c("pred1", "pred2"),
                            dependent = "outcome", unit.variable = "unit",
                            time.variable = "time", treatment.identifier = 1,
                            controls.identifier = 2:n_units, time.predictors.prior = 1:7,
                            time.optimize.ssr = 1:7, time.plot = 1:n_periods)
      Synth::synth(dp)
    }, slow = TRUE))
  }
}


# ============================================
# SPATIAL [n_side = 10, 20, 32]
# ============================================
cat("\n--- Spatial ---\n")

if (requireNamespace("spdep", quietly = TRUE) && requireNamespace("spatialreg", quietly = TRUE)) {
  for (n_side in c(10, 20, 32)) {
    n <- n_side^2
    set.seed(42)
    coords <- expand.grid(x = 0:(n_side-1), y = 0:(n_side-1))
    nb <- spdep::knn2nb(spdep::knearneigh(as.matrix(coords), k = 4))
    listw <- spdep::nb2listw(nb, style = "W")
    x_vals <- runif(n, -1, 1)
    y_vals <- 2.0 + 0.7 * x_vals + 0.3 * (coords$x + coords$y) / n_side + runif(n, -0.25, 0.25)
    sp_data <- data.frame(y = y_vals, x = x_vals)

    # SAR
    add_result(run_bench("SAR", "lagsarlm", n, function() spatialreg::lagsarlm(y ~ x, data = sp_data, listw = listw)))
    # SEM
    add_result(run_bench("SEM", "errorsarlm", n, function() spatialreg::errorsarlm(y ~ x, data = sp_data, listw = listw)))
    # SAC
    add_result(run_bench("SAC", "sacsarlm", n, function() spatialreg::sacsarlm(y ~ x, data = sp_data, listw = listw), slow = TRUE))
    # Moran
    add_result(run_bench("Moran_Test", "moran.test", n, function() spdep::moran.test(y_vals, listw, alternative = "greater")))
    # Local Moran (skip n_side=32)
    if (n_side <= 20) {
      add_result(run_bench("Local_Moran", "localmoran", n, function() spdep::localmoran(y_vals, listw), slow = TRUE))
    }
  }
}


# ============================================
# SURVIVAL [n = 100, 1000, 10000]
# ============================================
cat("\n--- Survival ---\n")

if (requireNamespace("survival", quietly = TRUE)) {
  for (n in c(100, 1000, 10000)) {
    sdata <- generate_survival_data(n)

    # Kaplan-Meier
    add_result(run_bench("KM", "unstratified", n, function() survival::survfit(survival::Surv(time, event) ~ 1, data = sdata)))
    # Cox PH
    add_result(run_bench("CoxPH", "efron", n, function() survival::coxph(survival::Surv(time, event) ~ x1 + x2, data = sdata)))
    # Log-Rank
    add_result(run_bench("LogRank", "test", n, function() survival::survdiff(survival::Surv(time, event) ~ group, data = sdata)))
    # AFT Weibull
    add_result(run_bench("AFT", "Weibull", n, function() survival::survreg(survival::Surv(time, event) ~ x1 + x2, data = sdata, dist = "weibull"), slow = TRUE))
  }
}


# ============================================
# STATISTICAL TESTS [n = 100, 1000, 10000]
# ============================================
cat("\n--- Statistical Tests ---\n")

for (n in c(100, 1000, 10000)) {
  set.seed(42)
  data_vec <- 5.0 + runif(n, -1, 1)

  # t-test
  add_result(run_bench("t_test", "one_sample", n, function() t.test(data_vec, mu = 5.0)))
  # Wilcoxon
  add_result(run_bench("Wilcoxon", "signed_rank", n, function() wilcox.test(data_vec, mu = 5.0)))
  # KS test
  add_result(run_bench("KS_test", "one_sample", n, function() ks.test(data_vec, "pnorm", mean = 5.0, sd = 1.0)))
  # Shapiro-Wilk (max 5000 in R)
  if (n <= 5000) {
    add_result(run_bench("Shapiro_Wilk", "test", n, function() shapiro.test(data_vec)))
  }
  # Cor_Test
  x_ct <- runif(n, -2, 2); y_ct <- 0.5 * x_ct + runif(n, -0.5, 0.5)
  add_result(run_bench("Cor_Test", "pearson", n, function() cor.test(x_ct, y_ct, method = "pearson")))
}

# ANOVA [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_per_group <- n / 3
  y_vals <- c(5.0 + runif(n_per_group, -1, 1),
              6.0 + runif(n_per_group, -1, 1),
              7.0 + runif(n - 2*n_per_group, -1, 1))
  group_vals <- c(rep("g0", n_per_group), rep("g1", n_per_group), rep("g2", n - 2*n_per_group))
  anova_df <- data.frame(y = y_vals, group = factor(group_vals))
  add_result(run_bench("ANOVA", "one_way", n, function() oneway.test(y ~ group, data = anova_df, var.equal = TRUE)))
}

# Kruskal-Wallis [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_per <- n / 3
  y_vals <- c(5.0 + runif(n_per, -1, 1), 6.0 + runif(n_per, -1, 1), 7.0 + runif(n - 2*n_per, -1, 1))
  group_vals <- c(rep("g0", n_per), rep("g1", n_per), rep("g2", n - 2*n_per))
  add_result(run_bench("Kruskal_Wallis", "test", n, function() kruskal.test(y_vals, factor(group_vals))))
}

# Friedman [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  treatments <- matrix(c(5.0 + runif(n, -1, 1), 6.0 + runif(n, -1, 1), 7.0 + runif(n, -1, 1)), nrow = n, ncol = 3)
  add_result(run_bench("Friedman", "test", n, function() friedman.test(treatments)))
}

# Chi-squared [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  observed <- (n / 5) + runif(5, -5, 5)
  add_result(run_bench("Chi_squared", "gof", n, function() chisq.test(observed)))
}

# Fisher [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  a <- round(n * 0.3); b <- round(n * 0.2); cc <- round(n * 0.15); d <- n - a - b - cc
  add_result(run_bench("Fisher", "twosided", n, function() fisher.test(matrix(c(a, cc, b, d), nrow = 2))))
}

# Bartlett [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5); g3 <- runif(n, -2, 2)
  all_vals <- c(g1, g2, g3)
  groups <- factor(rep(c("g1","g2","g3"), each = n))
  add_result(run_bench("Bartlett", "test", n, function() bartlett.test(all_vals, groups)))
}

# Var_Test [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5)
  add_result(run_bench("Var_Test", "F-test", n, function() var.test(g1, g2)))
}

# McNemar [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  a <- round(n * 0.4); b <- round(n * 0.15); cc <- round(n * 0.25); d <- n - a - b - cc
  add_result(run_bench("McNemar", "test", n, function() mcnemar.test(matrix(c(a, cc, b, d), nrow = 2))))
}

# Tukey HSD [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_per <- n / 3
  values <- c(runif(n_per, -1, 1), 0.5 + runif(n_per, -1, 1), 1.0 + runif(n - 2*n_per, -1, 1))
  group_labels <- factor(c(rep("g0", n_per), rep("g1", n_per), rep("g2", n - 2*n_per)))
  tk_df <- data.frame(y = values, group = group_labels)
  add_result(run_bench("Tukey", "HSD", n, function() TukeyHSD(aov(y ~ group, data = tk_df))))
}

# Prop_Test [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  x <- round(n * 0.55)
  add_result(run_bench("Prop_Test", "one-sample", n, function() prop.test(x, n, p = 0.5)))
}

# Binom_Test [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  x <- round(n * 0.55)
  add_result(run_bench("Binom_Test", "exact", n, function() binom.test(x, n, p = 0.5)))
}

# Poisson_Test [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  add_result(run_bench("Poisson_Test", "exact", n, function() poisson.test(n, T = 1, r = n)))
}

# Isotonic Regression [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  x <- (0:(n-1)) / n
  y_iso <- x * 2 + runif(n, -0.5, 0.5)
  add_result(run_bench("Isotonic_Regression", "PAVA", n, function() isoreg(y_iso)))
}

# Canonical Correlation [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  data <- generate_regression_data(n)
  xmat <- as.matrix(data[, c("x1", "x2")])
  ymat <- as.matrix(data[, c("x3", "x4", "x5")])
  add_result(run_bench("Cancor", "canonical", n, function() cancor(xmat, ymat)))
}

# Spline [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  x <- (0:(n-1)) / n
  y_sp <- sin(x * 2 * pi) + runif(n, -0.1, 0.1)
  add_result(run_bench("Spline", "natural", n, function() spline(x, y_sp, n = n * 3, method = "natural")))
}

# MANOVA [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_groups <- 3; n_per <- n / n_groups
  y1 <- c(1.5*0 + runif(n_per,-1,1), 1.5*1 + runif(n_per,-1,1), 1.5*2 + runif(n-2*n_per,-1,1))
  y2 <- c(0.8*0 + runif(n_per,-1,1), 0.8*1 + runif(n_per,-1,1), 0.8*2 + runif(n-2*n_per,-1,1))
  grp <- factor(c(rep("g0",n_per), rep("g1",n_per), rep("g2",n-2*n_per)))
  add_result(run_bench("MANOVA", "one-way", n, function() summary(manova(cbind(y1, y2) ~ grp))))
}

# Power Analysis [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  add_result(run_bench("Power_Analysis", "t-test", n, function() power.t.test(n = n, delta = 0.5, sd = 1, sig.level = 0.05, type = "two.sample")))
}

# Mahalanobis [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  mdata <- matrix(runif(n * 3, -2, 2), nrow = n, ncol = 3)
  add_result(run_bench("Mahalanobis", "distances", n, function() {
    cm <- colMeans(mdata); cv <- cov(mdata)
    mahalanobis(mdata, cm, cv)
  }))
}

# Median Polish [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  rows <- as.integer(sqrt(n)); cols <- rows
  mat <- matrix(0, nrow = rows, ncol = cols)
  for (r in 1:rows) for (c_idx in 1:cols) mat[r, c_idx] <- (r-1)*0.5 + (c_idx-1)*0.3 + runif(1, -0.5, 0.5)
  add_result(run_bench("Median_Polish", "iterative", n, function() medpolish(mat, trace.iter = FALSE)))
}

# Quade [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  treatments <- matrix(c(0*0.3 + runif(n, -1, 1), 1*0.3 + runif(n, -1, 1), 2*0.3 + runif(n, -1, 1)), nrow = n, ncol = 3)
  groups <- gl(3, 1, n * 3)
  blocks <- gl(n, 3)
  quade_data <- data.frame(y = as.vector(treatments), group = groups, block = blocks)
  add_result(run_bench("Quade", "test", n, function() {
    friedman.test(y ~ group | block, data = quade_data)  # R has no quade.test, Friedman is similar
  }))
}

# P_Adjust [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  pvals <- runif(n)
  add_result(run_bench("P_Adjust", "BH", n, function() p.adjust(pvals, method = "BH")))
}

# Robust Stats (fivenum + IQR) [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  data_vec <- runif(n, -5, 5)
  add_result(run_bench("Robust_Stats", "fivenum+iqr", n, function() { fivenum(data_vec); IQR(data_vec) }))
}

# Weighted Mean [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  data_vec <- runif(n, -5, 5)
  weights <- runif(n, 0.1, 2)
  add_result(run_bench("Weighted", "mean", n, function() weighted.mean(data_vec, weights)))
}

# Loglin [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  tbl <- array(runif(8, 5, 50), dim = c(2,2,2))
  add_result(run_bench("Loglin", "3-way", n, function() loglin(tbl, margin = list(c(1,2), c(2,3)), print = FALSE)))
}


# Fligner [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5); g3 <- runif(n, -2, 2)
  all_vals <- c(g1, g2, g3)
  groups <- factor(rep(c("g1","g2","g3"), each = n))
  add_result(run_bench("Fligner", "test", n, function() fligner.test(all_vals, groups)))
}

# Mood [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5)
  add_result(run_bench("Mood", "test", n, function() mood.test(g1, g2)))
}

# Ansari [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5)
  add_result(run_bench("Ansari", "test", n, function() ansari.test(g1, g2)))
}

# Oneway (Welch ANOVA) [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  g1 <- runif(n, -1, 1); g2 <- runif(n, -0.5, 1.5); g3 <- runif(n, -2, 2)
  all_vals <- c(g1, g2, g3)
  groups <- factor(rep(c("g1","g2","g3"), each = n))
  add_result(run_bench("Oneway", "Welch", n, function() oneway.test(all_vals ~ groups, var.equal = FALSE)))
}

# Mantel-Haenszel [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_strata <- max(n / 25, 4)
  tbl <- array(0, dim = c(2, 2, n_strata))
  for (s in 1:n_strata) {
    tbl[1,1,s] <- sample(5:20, 1)
    tbl[1,2,s] <- sample(3:15, 1)
    tbl[2,1,s] <- sample(3:15, 1)
    tbl[2,2,s] <- sample(5:20, 1)
  }
  add_result(run_bench("Mantel_Haenszel", "CMH", n, function() mantelhaen.test(tbl)))
}

# Pairwise_t [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_per <- n / 3
  values <- c(runif(n_per, -1, 1), 0.5 + runif(n_per, -1, 1), 1.0 + runif(n - 2*n_per, -1, 1))
  group_labels <- factor(c(rep("g0", n_per), rep("g1", n_per), rep("g2", n - 2*n_per)))
  add_result(run_bench("Pairwise_t", "Holm", n, function() pairwise.t.test(values, group_labels, p.adjust.method = "holm")))
}

# Pairwise_Wilcox [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  set.seed(42)
  n_per <- n / 3
  values <- c(runif(n_per, -1, 1), 0.5 + runif(n_per, -1, 1), 1.0 + runif(n - 2*n_per, -1, 1))
  group_labels <- factor(c(rep("g0", n_per), rep("g1", n_per), rep("g2", n - 2*n_per)))
  add_result(run_bench("Pairwise_Wilcox", "Holm", n, function() pairwise.wilcox.test(values, group_labels, p.adjust.method = "holm")))
}

# Prop_Trend [n = 100, 1000, 10000]
for (n in c(100, 1000, 10000)) {
  k <- 5
  x_vals <- sapply(0:(k-1), function(i) (n / k) / 2 + i * 3)
  n_vals <- rep(n / k, k)
  add_result(run_bench("Prop_Trend", "test", n, function() prop.trend.test(x_vals, n_vals)))
}

# Staggered DiD [n = 100, 1000] (if did package available)
if (requireNamespace("did", quietly = TRUE)) {
  for (params in list(c(10, 10), c(50, 20))) {
    n_units <- params[1]; n_periods <- params[2]; n <- n_units * n_periods
    sdata <- generate_staggered_panel(n_units, n_periods)
    sdata$unit <- sdata$unit + 1  # did package expects 1-based
    sdata$time <- sdata$time + 1
    # treat_time=0 means never-treated; shift to large number for did::att_gt
    sdata$treat_time[sdata$treat_time == 0] <- 0  # did uses 0 for never-treated
    add_result(run_bench("Staggered_DiD", "CS", n, function() {
      did::att_gt(yname = "y", tname = "time", idname = "unit", gname = "treat_time", data = sdata)
    }, slow = TRUE))
  }
}

# Mediation [n = 100, 1000] (if mediation package available)
if (requireNamespace("mediation", quietly = TRUE)) {
  for (n in c(100, 1000)) {
    set.seed(42)
    x1 <- runif(n, -1, 1)
    treatment <- rbinom(n, 1, 0.5)
    mediator <- 0.5 * treatment + 0.3 * x1 + runif(n, -0.5, 0.5)
    y <- 1.0 + 0.3 * treatment + 0.5 * mediator + 0.2 * x1 + runif(n, -0.5, 0.5)
    med_data <- data.frame(y = y, treatment = treatment, mediator = mediator, x1 = x1)
    add_result(run_bench("Mediation", "IPW", n, function() {
      m_model <- lm(mediator ~ treatment + x1, data = med_data)
      y_model <- lm(y ~ treatment + mediator + x1, data = med_data)
      mediation::mediate(m_model, y_model, treat = "treatment", mediator = "mediator", sims = 199)
    }, slow = TRUE))
  }
}


# ============================================
# Save Results
# ============================================

cat("\n\n--- Saving results ---\n")

# Filter NULLs
results <- results[!sapply(results, is.null)]

if (length(results) == 0) {
  cat("No benchmark results to save.\n")
  quit(status = 0)
}

results_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(
    method = r$method, variant = r$variant, n = r$n,
    iterations = r$iterations,
    time_min_us = r$time_min_us, time_p25_us = r$time_p25_us,
    time_median_us = r$time_median_us, time_p75_us = r$time_p75_us,
    time_max_us = r$time_max_us, time_mean_us = r$time_mean_us,
    time_std_us = r$time_std_us, itr_per_sec = r$itr_per_sec,
    mem_alloc_bytes = r$mem_alloc_bytes,
    stringsAsFactors = FALSE
  )
}))

dir.create("results", showWarnings = FALSE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_file <- sprintf("results/r_unified_%s.csv", timestamp)
write.csv(results_df, output_file, row.names = FALSE)

cat(sprintf("Results saved to: %s\n", output_file))
cat(sprintf("Total benchmarks: %d\n", nrow(results_df)))
cat(sprintf("R version: %s\n", R.version.string))
