#!/usr/bin/env Rscript
# Data Munging Benchmarks for Cross-Language Comparison
# Compares R's data.table and dplyr against p2a Rust/Polars implementation

library(microbenchmark)
library(data.table)
library(dplyr)
library(tidyr)

# Ensure reproducibility
set.seed(42)

# =============================================================================
# DATA GENERATION
# =============================================================================

# Generate standard munging data (matches Rust benchmarks)
generate_munging_data <- function(n) {
  data.table(
    id = 1:n,
    group = sample(1:100, n, replace = TRUE),
    x1 = runif(n, -1, 1),
    x2 = runif(n, -1, 1),
    x3 = runif(n, -1, 1),
    category = sample(letters[1:10], n, replace = TRUE)
  )
}

# Generate join data (left table + right lookup table at 1/10 size)
generate_join_data <- function(n) {
  left <- generate_munging_data(n)

  right_n <- n / 10
  right <- data.table(
    group = 1:right_n,
    lookup_value = runif(right_n, 0, 100),
    lookup_label = paste0("label_", seq_len(right_n) %% 100)
  )

  list(left = left, right = right)
}

# Generate panel data for lag/lead benchmarks
generate_panel_data <- function(n_entities, n_periods) {
  n <- n_entities * n_periods
  data.table(
    entity = rep(1:n_entities, each = n_periods),
    period = rep(1:n_periods, n_entities),
    value = runif(n, -1, 1)
  )
}

# Generate pivot data (long format)
generate_pivot_data <- function(n_ids, n_vars) {
  n <- n_ids * n_vars
  data.table(
    id = rep(1:n_ids, each = n_vars),
    variable = rep(paste0("var_", 1:n_vars), n_ids),
    value = runif(n, 0, 100)
  )
}

# Generate data with NAs
generate_data_with_na <- function(n, na_fraction = 0.1) {
  data.table(
    id = 1:n,
    x1 = ifelse(runif(n) < na_fraction, NA_real_, runif(n, -1, 1)),
    x2 = ifelse(runif(n) < na_fraction, NA_real_, runif(n, -1, 1))
  )
}

# =============================================================================
# FILTER BENCHMARKS
# =============================================================================

benchmark_filter <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking filter with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    # Numeric filter (x1 > 0)
    bm_numeric <- microbenchmark(
      data.table = dt[x1 > 0],
      dplyr = filter(df, x1 > 0),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("filter_numeric_", n)]] <- summary(bm_numeric)

    # String filter (category == "a")
    bm_string <- microbenchmark(
      data.table = dt[category == "a"],
      dplyr = filter(df, category == "a"),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("filter_string_", n)]] <- summary(bm_string)
  }

  results
}

# =============================================================================
# SELECT BENCHMARKS
# =============================================================================

benchmark_select <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking select with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    bm <- microbenchmark(
      data.table = dt[, .(id, x1, x2)],
      dplyr = select(df, id, x1, x2),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("select_", n)]] <- summary(bm)
  }

  results
}

# =============================================================================
# SORT BENCHMARKS
# =============================================================================

benchmark_sort <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking sort with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    # Single column sort
    bm_single <- microbenchmark(
      data.table = {
        dt_copy <- copy(dt)
        setorder(dt_copy, x1)
      },
      dplyr = arrange(df, x1),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("sort_single_", n)]] <- summary(bm_single)

    # Multi-column sort
    bm_multi <- microbenchmark(
      data.table = {
        dt_copy <- copy(dt)
        setorder(dt_copy, group, -x1)
      },
      dplyr = arrange(df, group, desc(x1)),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("sort_multi_", n)]] <- summary(bm_multi)
  }

  results
}

# =============================================================================
# JOIN BENCHMARKS
# =============================================================================

benchmark_join <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking left join with n=%d\n", n))
    data <- generate_join_data(n)
    dt_left <- data$left
    dt_right <- data$right
    df_left <- as.data.frame(dt_left)
    df_right <- as.data.frame(dt_right)

    bm <- microbenchmark(
      data.table = merge(dt_left, dt_right, by = "group", all.x = TRUE),
      dplyr = left_join(df_left, df_right, by = "group"),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("left_join_", n)]] <- summary(bm)
  }

  results
}

# =============================================================================
# GROUP BY BENCHMARKS
# =============================================================================

benchmark_group_by <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking group_by with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    # Single aggregation
    bm_single_agg <- microbenchmark(
      data.table = dt[, .(sum_x1 = sum(x1)), by = group],
      dplyr = df %>% group_by(group) %>% summarize(sum_x1 = sum(x1)),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("group_by_single_agg_", n)]] <- summary(bm_single_agg)

    # Multiple aggregations
    bm_multi_agg <- microbenchmark(
      data.table = dt[, .(sum_x1 = sum(x1), mean_x2 = mean(x2), max_x3 = max(x3)), by = group],
      dplyr = df %>% group_by(group) %>% summarize(sum_x1 = sum(x1), mean_x2 = mean(x2), max_x3 = max(x3)),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("group_by_multi_agg_", n)]] <- summary(bm_multi_agg)

    # Multiple keys
    bm_multi_key <- microbenchmark(
      data.table = dt[, .(sum_x1 = sum(x1), mean_x2 = mean(x2)), by = .(group, category)],
      dplyr = df %>% group_by(group, category) %>% summarize(sum_x1 = sum(x1), mean_x2 = mean(x2), .groups = "drop"),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("group_by_multi_key_", n)]] <- summary(bm_multi_key)
  }

  results
}

# =============================================================================
# PIVOT/MELT BENCHMARKS
# =============================================================================

benchmark_reshape <- function() {
  results <- list()

  # Pivot (long to wide)
  for (n_ids in c(1000, 10000, 100000)) {
    cat(sprintf("Benchmarking pivot with n_ids=%d\n", n_ids))
    dt <- generate_pivot_data(n_ids, 5)
    df <- as.data.frame(dt)

    bm_pivot <- microbenchmark(
      data.table = dcast(dt, id ~ variable, value.var = "value"),
      dplyr = pivot_wider(df, id_cols = id, names_from = variable, values_from = value),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("pivot_", n_ids)]] <- summary(bm_pivot)
  }

  # Melt (wide to long)
  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking melt with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    bm_melt <- microbenchmark(
      data.table = melt(dt, id.vars = c("id", "group"), measure.vars = c("x1", "x2", "x3")),
      dplyr = pivot_longer(df, cols = c(x1, x2, x3), names_to = "variable", values_to = "value"),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("melt_", n)]] <- summary(bm_melt)
  }

  results
}

# =============================================================================
# LAG/LEAD BENCHMARKS
# =============================================================================

benchmark_lag <- function() {
  results <- list()

  # Simple lag (no grouping)
  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking simple lag with n=%d\n", n))
    dt <- generate_munging_data(n)
    df <- as.data.frame(dt)

    bm_simple <- microbenchmark(
      data.table = dt[, x1_lag := shift(x1, 1)],
      dplyr = mutate(df, x1_lag = lag(x1, 1)),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("lag_simple_", n)]] <- summary(bm_simple)
  }

  # Grouped lag (panel data)
  for (config in list(c(100, 100), c(500, 200), c(1000, 1000))) {
    n_entities <- config[1]
    n_periods <- config[2]
    label <- paste0(n_entities, "x", n_periods)
    cat(sprintf("Benchmarking grouped lag with %s\n", label))

    dt <- generate_panel_data(n_entities, n_periods)
    df <- as.data.frame(dt)

    bm_grouped <- microbenchmark(
      data.table = dt[, value_lag := shift(value, 1), by = entity],
      dplyr = df %>% group_by(entity) %>% mutate(value_lag = lag(value, 1)) %>% ungroup(),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("lag_grouped_", label)]] <- summary(bm_grouped)
  }

  results
}

# =============================================================================
# FILL NA BENCHMARKS
# =============================================================================

benchmark_fill_na <- function() {
  results <- list()

  for (n in c(10000, 100000, 1000000)) {
    cat(sprintf("Benchmarking fill_na with n=%d\n", n))
    dt <- generate_data_with_na(n, 0.1)
    df <- as.data.frame(dt)

    # Forward fill
    bm_forward <- microbenchmark(
      data.table = {
        dt_copy <- copy(dt)
        setnafill(dt_copy, type = "locf", cols = c("x1", "x2"))
      },
      dplyr = df %>% fill(x1, x2, .direction = "down"),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("fill_na_forward_", n)]] <- summary(bm_forward)

    # Mean fill (data.table uses column mean, dplyr uses replace_na)
    bm_mean <- microbenchmark(
      data.table = {
        dt_copy <- copy(dt)
        dt_copy[, x1 := fifelse(is.na(x1), mean(x1, na.rm = TRUE), x1)]
        dt_copy[, x2 := fifelse(is.na(x2), mean(x2, na.rm = TRUE), x2)]
      },
      dplyr = df %>%
        mutate(
          x1 = ifelse(is.na(x1), mean(x1, na.rm = TRUE), x1),
          x2 = ifelse(is.na(x2), mean(x2, na.rm = TRUE), x2)
        ),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("fill_na_mean_", n)]] <- summary(bm_mean)
  }

  results
}

# =============================================================================
# RUN ALL BENCHMARKS
# =============================================================================

cat("=== Filter Benchmarks ===\n")
filter_results <- benchmark_filter()

cat("\n=== Select Benchmarks ===\n")
select_results <- benchmark_select()

cat("\n=== Sort Benchmarks ===\n")
sort_results <- benchmark_sort()

cat("\n=== Join Benchmarks ===\n")
join_results <- benchmark_join()

cat("\n=== Group By Benchmarks ===\n")
groupby_results <- benchmark_group_by()

cat("\n=== Reshape Benchmarks ===\n")
reshape_results <- benchmark_reshape()

cat("\n=== Lag Benchmarks ===\n")
lag_results <- benchmark_lag()

cat("\n=== Fill NA Benchmarks ===\n")
fillna_results <- benchmark_fill_na()

# =============================================================================
# SAVE RESULTS
# =============================================================================

save_results <- function(results, filename) {
  df <- do.call(rbind, lapply(names(results), function(name) {
    r <- results[[name]]
    data.frame(
      method = name,
      expr = as.character(r$expr),
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

save_results(filter_results, "results/munging_filter.csv")
save_results(select_results, "results/munging_select.csv")
save_results(sort_results, "results/munging_sort.csv")
save_results(join_results, "results/munging_join.csv")
save_results(groupby_results, "results/munging_groupby.csv")
save_results(reshape_results, "results/munging_reshape.csv")
save_results(lag_results, "results/munging_lag.csv")
save_results(fillna_results, "results/munging_fillna.csv")

# Print summary
cat("\n=== Summary ===\n")
cat("Filter results:\n")
print(do.call(rbind, filter_results))
cat("\nGroup By results:\n")
print(do.call(rbind, groupby_results))
cat("\nJoin results:\n")
print(do.call(rbind, join_results))
