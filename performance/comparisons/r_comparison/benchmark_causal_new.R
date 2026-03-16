# Benchmark new causal inference methods: MatchIt, WeightIt, bacondecomp, sensemakr, TMLE, CBPS
# Compare R implementations to Rust

library(microbenchmark)

# Check and load packages
packages <- c("MatchIt", "WeightIt", "bacondecomp", "sensemakr", "tmle", "CBPS", "did")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Package %s not installed, skipping...\n", pkg))
  }
}

set.seed(42)

# Generate test data for causal inference
n <- 10000
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)

# Treatment assignment (depends on covariates)
ps <- plogis(0.5 + 0.3*x1 + 0.2*x2 - 0.1*x3)
treatment <- rbinom(n, 1, ps)

# Outcome (with treatment effect)
y <- 2 + 0.5*treatment + 0.3*x1 + 0.2*x2 + 0.1*x3 + rnorm(n)

df <- data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2, x3 = x3)

results <- list()

# 1. MatchIt benchmark
if (requireNamespace("MatchIt", quietly = TRUE)) {
  cat("\n=== MatchIt (Propensity Score Matching) ===\n")

  mb_matchit <- microbenchmark(
    matchit_nn = MatchIt::matchit(treatment ~ x1 + x2 + x3, data = df, method = "nearest"),
    matchit_cem = MatchIt::matchit(treatment ~ x1 + x2 + x3, data = df, method = "cem"),
    times = 20
  )
  print(mb_matchit)
  results$matchit <- summary(mb_matchit)
}

# 2. WeightIt benchmark
if (requireNamespace("WeightIt", quietly = TRUE)) {
  cat("\n=== WeightIt (IPW Weighting) ===\n")

  mb_weightit <- microbenchmark(
    weightit_ps = WeightIt::weightit(treatment ~ x1 + x2 + x3, data = df, method = "ps"),
    weightit_ebal = WeightIt::weightit(treatment ~ x1 + x2 + x3, data = df, method = "ebal"),
    times = 20
  )
  print(mb_weightit)
  results$weightit <- summary(mb_weightit)
}

# 3. CBPS benchmark
if (requireNamespace("CBPS", quietly = TRUE)) {
  cat("\n=== CBPS (Covariate Balancing Propensity Score) ===\n")

  mb_cbps <- microbenchmark(
    cbps = CBPS::CBPS(treatment ~ x1 + x2 + x3, data = df, ATT = FALSE),
    times = 10
  )
  print(mb_cbps)
  results$cbps <- summary(mb_cbps)
}

# 4. sensemakr benchmark
if (requireNamespace("sensemakr", quietly = TRUE)) {
  cat("\n=== sensemakr (Sensitivity Analysis) ===\n")

  # First fit OLS
  model <- lm(y ~ treatment + x1 + x2 + x3, data = df)

  mb_sensemakr <- microbenchmark(
    sensemakr = sensemakr::sensemakr(model, treatment = "treatment", benchmark_covariates = "x1"),
    times = 50
  )
  print(mb_sensemakr)
  results$sensemakr <- summary(mb_sensemakr)
}

# 5. TMLE benchmark
if (requireNamespace("tmle", quietly = TRUE)) {
  cat("\n=== TMLE (Targeted Maximum Likelihood) ===\n")

  W <- as.matrix(df[, c("x1", "x2", "x3")])

  mb_tmle <- microbenchmark(
    tmle = tmle::tmle(Y = df$y, A = df$treatment, W = W),
    times = 5
  )
  print(mb_tmle)
  results$tmle <- summary(mb_tmle)
}

# 6. marginaleffects benchmark
if (requireNamespace("marginaleffects", quietly = TRUE)) {
  cat("\n=== marginaleffects (Average Marginal Effects) ===\n")

  model <- lm(y ~ treatment + x1 + x2 + x3, data = df)

  mb_mfx <- microbenchmark(
    marginaleffects = marginaleffects::avg_slopes(model),
    times = 50
  )
  print(mb_mfx)
  results$marginaleffects <- summary(mb_mfx)
}

# 7. Bacon decomposition (need panel data)
if (requireNamespace("bacondecomp", quietly = TRUE)) {
  cat("\n=== bacondecomp (Goodman-Bacon Decomposition) ===\n")

  # Create panel data with staggered treatment
  n_units <- 100
  n_periods <- 100
  panel <- expand.grid(unit = 1:n_units, time = 1:n_periods)
  panel$treat_time <- ifelse(panel$unit <= 15, 4,
                              ifelse(panel$unit <= 30, 6,
                                     ifelse(panel$unit <= 40, 8, Inf)))
  panel$treated <- as.numeric(panel$time >= panel$treat_time)
  panel$y <- 1 + 0.5 * panel$treated + 0.1 * panel$time + rnorm(nrow(panel), sd = 0.5)

  mb_bacon <- microbenchmark(
    bacon = bacondecomp::bacon(y ~ treated, data = panel, id_var = "unit", time_var = "time"),
    times = 10
  )
  print(mb_bacon)
  results$bacon <- summary(mb_bacon)
}

# Summary
cat("\n\n========================================\n")
cat("R BENCHMARK SUMMARY (milliseconds)\n")
cat("========================================\n\n")

for (name in names(results)) {
  res <- results[[name]]
  cat(sprintf("%s:\n", toupper(name)))
  for (i in 1:nrow(res)) {
    cat(sprintf("  %s: median=%.2f ms, mean=%.2f ms\n",
                res$expr[i], res$median[i]/1e6, res$mean[i]/1e6))
  }
  cat("\n")
}

# Save results to CSV
df_results <- do.call(rbind, lapply(names(results), function(name) {
  res <- results[[name]]
  data.frame(
    method = name,
    variant = as.character(res$expr),
    median_ms = res$median / 1e6,
    mean_ms = res$mean / 1e6,
    min_ms = res$min / 1e6,
    max_ms = res$max / 1e6
  )
}))

write.csv(df_results, "r_causal_benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to r_causal_benchmark_results.csv\n")
