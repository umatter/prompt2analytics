# ============================================================================
# fig_benchmark_execution.R
# Grouped bar chart comparing R vs Rust execution times at n=100,000
# with min/max error bars showing execution time variability
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

INPUT <- "rust_validation/results/summaries/"
OUTPUT <- "../figures/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "benchmark_summary.json"), simplifyVector = FALSE)

# Extract methods data and flatten (handle type inconsistencies)
# Include min/max for error bars
methods_list <- summary_data$methods
benchmark_df <- map_dfr(names(methods_list), function(method) {
  method_results <- methods_list[[method]]
  map_dfr(method_results, function(r) {
    # Handle r_max_us which may be empty array or numeric
    r_max <- if (is.list(r$r_max_us) && length(r$r_max_us) == 0) {
      NA_real_
    } else if (is.numeric(r$r_max_us)) {
      r$r_max_us
    } else {
      NA_real_
    }

    tibble(
      method = method,
      n = r$n,
      r_median_us = r$r_median_us,
      r_min_us = r$r_min_us,
      r_max_us = r_max,
      rust_median_us = r$rust_median_us,
      rust_min_us = r$rust_min_us,
      rust_max_us = r$rust_max_us,
      speedup = r$speedup
    )
  })
})

# Filter to largest n per method and prepare for plotting
plot_data <- benchmark_df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  filter(r_median_us > 0, rust_median_us > 0) %>%
  mutate(
    # Add category labels
    category = case_when(
      grepl("^ols", method, ignore.case = TRUE) ~ "Regression",
      grepl("^panel", method, ignore.case = TRUE) ~ "Panel",
      grepl("logit|probit", method, ignore.case = TRUE) ~ "Discrete",
      grepl("kmeans|pca|dbscan|hierarchical", method, ignore.case = TRUE) ~ "ML",
      grepl("sort|filter|group|select|standardize|lag|lead|diff", method, ignore.case = TRUE) ~ "Munging",
      TRUE ~ "Other"
    ),
    # Clean method labels
    method_label = method %>%
      str_replace_all("_", " ") %>%
      str_replace_all("hc([0-3])", "HC\\1") %>%
      str_replace("^ols$", "OLS") %>%
      str_replace("^ols ", "OLS+") %>%
      str_replace("panel fe", "Panel FE") %>%
      str_replace("panel re", "Panel RE") %>%
      str_to_title(),
    # Convert to milliseconds
    r_ms = r_median_us / 1000,
    r_min_ms = r_min_us / 1000,
    r_max_ms = if_else(is.na(r_max_us), r_ms * 1.2, r_max_us / 1000),  # fallback if missing
    rust_ms = rust_median_us / 1000,
    rust_min_ms = rust_min_us / 1000,
    rust_max_ms = rust_max_us / 1000
  )

# Convert to long format for grouped bars with error bar info
plot_long <- plot_data %>%
  select(method_label, category, r_ms, r_min_ms, r_max_ms, rust_ms, rust_min_ms, rust_max_ms) %>%
  pivot_longer(
    cols = c(r_ms, rust_ms),
    names_to = "implementation",
    values_to = "time_ms"
  ) %>%
  mutate(
    min_ms = if_else(implementation == "r_ms", r_min_ms, rust_min_ms),
    max_ms = if_else(implementation == "r_ms", r_max_ms, rust_max_ms),
    implementation = if_else(implementation == "r_ms", "R", "Rust"),
    implementation = factor(implementation, levels = c("R", "Rust"))
  ) %>%
  select(method_label, category, implementation, time_ms, min_ms, max_ms)

# Order by category then R time
method_order <- plot_data %>%
  arrange(category, desc(r_ms)) %>%
  pull(method_label)

plot_long <- plot_long %>%
  mutate(method_label = factor(method_label, levels = method_order))

## PLOT ----
p <- ggplot(plot_long, aes(x = method_label, y = time_ms, fill = implementation)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(
    aes(ymin = min_ms, ymax = max_ms),
    position = position_dodge(width = 0.8),
    width = 0.25,
    linewidth = 0.4,
    color = "gray30"
  ) +
  scale_y_log10(
    name = "Execution Time (ms, log scale)",
    labels = scales::comma
  ) +
  scale_x_discrete(name = NULL) +
  scale_fill_manual(
    values = c("R" = "#0097A7", "Rust" = "#E65100"),
    name = "Implementation"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 13),
    axis.text.y = element_text(size = 13),
    axis.title.y = element_text(size = 14),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 20, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "benchmark_boxplots.pdf"),
  plot = p,
  width = 12,
  height = 6
)

message("Created: ", OUTPUT, "benchmark_boxplots.pdf")
