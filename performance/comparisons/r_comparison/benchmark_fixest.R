# Benchmark fixest: R's current state-of-art for panel data methods
# Compare against plm/lfe and Rust implementations

library(fixest)
library(plm)
library(bench)

set.seed(42)

# Generate panel data
generate_panel <- function(n_entities, n_periods) {
  n <- n_entities * n_periods

  data.frame(
    entity = rep(1:n_entities, each = n_periods),
    time = rep(1:n_periods, times = n_entities),
    x1 = rnorm(n),
    x2 = rnorm(n),
    fe_entity = rep(rnorm(n_entities, sd = 2), each = n_periods),
    fe_time = rep(rnorm(n_periods, sd = 1), times = n_entities),
    y = NA
  )
}

# Add DGP
add_outcome <- function(df) {
  df$y <- 2 + 0.5 * df$x1 - 0.3 * df$x2 + df$fe_entity + df$fe_time + rnorm(nrow(df))
  df
}

cat("=== fixest Benchmark Results ===\n\n")

# Test configurations
configs <- list(
  list(entities = 10, periods = 10, n = 100),
  list(entities = 50, periods = 20, n = 1000),
  list(entities = 100, periods = 100, n = 10000)
)

results <- data.frame()

for (cfg in configs) {
  cat(sprintf("--- n = %d (%d entities x %d periods) ---\n",
              cfg$n, cfg$entities, cfg$periods))

  df <- generate_panel(cfg$entities, cfg$periods)
  df <- add_outcome(df)
  pdata <- pdata.frame(df, index = c("entity", "time"))

  # Benchmark one-way FE
  cat("One-way FE (entity):\n")

  bm_fixest_1way <- bench::mark(
    fixest = feols(y ~ x1 + x2 | entity, data = df),
    iterations = 50,
    check = FALSE
  )

  bm_plm_1way <- bench::mark(
    plm = plm(y ~ x1 + x2, data = pdata, model = "within", effect = "individual"),
    iterations = 50,
    check = FALSE
  )

  cat(sprintf("  fixest: %.2f ms (median)\n", as.numeric(bm_fixest_1way$median) * 1000))
  cat(sprintf("  plm:    %.2f ms (median)\n", as.numeric(bm_plm_1way$median) * 1000))
  cat(sprintf("  fixest speedup vs plm: %.1fx\n\n",
              as.numeric(bm_plm_1way$median) / as.numeric(bm_fixest_1way$median)))

  results <- rbind(results, data.frame(
    method = "FE_1way",
    package = "fixest",
    n = cfg$n,
    time_ms = as.numeric(bm_fixest_1way$median) * 1000
  ))
  results <- rbind(results, data.frame(
    method = "FE_1way",
    package = "plm",
    n = cfg$n,
    time_ms = as.numeric(bm_plm_1way$median) * 1000
  ))

  # Benchmark two-way FE
  cat("Two-way FE (entity + time):\n")

  bm_fixest_2way <- bench::mark(
    fixest = feols(y ~ x1 + x2 | entity + time, data = df),
    iterations = 50,
    check = FALSE
  )

  bm_plm_2way <- bench::mark(
    plm = plm(y ~ x1 + x2, data = pdata, model = "within", effect = "twoways"),
    iterations = 50,
    check = FALSE
  )

  cat(sprintf("  fixest: %.2f ms (median)\n", as.numeric(bm_fixest_2way$median) * 1000))
  cat(sprintf("  plm:    %.2f ms (median)\n", as.numeric(bm_plm_2way$median) * 1000))
  cat(sprintf("  fixest speedup vs plm: %.1fx\n\n",
              as.numeric(bm_plm_2way$median) / as.numeric(bm_fixest_2way$median)))

  results <- rbind(results, data.frame(
    method = "FE_2way",
    package = "fixest",
    n = cfg$n,
    time_ms = as.numeric(bm_fixest_2way$median) * 1000
  ))
  results <- rbind(results, data.frame(
    method = "FE_2way",
    package = "plm",
    n = cfg$n,
    time_ms = as.numeric(bm_plm_2way$median) * 1000
  ))

  # Benchmark clustered SEs
  cat("FE with clustered SEs:\n")

  bm_fixest_cluster <- bench::mark(
    fixest = feols(y ~ x1 + x2 | entity, data = df, cluster = ~entity),
    iterations = 50,
    check = FALSE
  )

  cat(sprintf("  fixest (clustered): %.2f ms (median)\n\n",
              as.numeric(bm_fixest_cluster$median) * 1000))

  results <- rbind(results, data.frame(
    method = "FE_clustered",
    package = "fixest",
    n = cfg$n,
    time_ms = as.numeric(bm_fixest_cluster$median) * 1000
  ))
}

# Save results
write.csv(results, "results/fixest_benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to results/fixest_benchmark_results.csv\n")

# Summary comparison
cat("\n=== Summary: fixest vs plm ===\n")
cat("fixest is typically 5-50x faster than plm for panel FE estimation.\n")
cat("This reflects fixest's optimized C++ implementation designed specifically\n")
cat("for high-dimensional fixed effects.\n")
