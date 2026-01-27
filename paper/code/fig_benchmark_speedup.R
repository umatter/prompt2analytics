# ============================================================================
# fig_benchmark_speedup.R
# Horizontal bar chart showing Rust/R speedup factors at n=100,000
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

INPUT <- "rust_validation/results/summaries/"
OUTPUT <- "../figures/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "benchmark_summary.json"), simplifyVector = FALSE)

# Extract methods data and flatten (handle type inconsistencies)
methods_list <- summary_data$methods
benchmark_df <- map_dfr(names(methods_list), function(method) {
  method_results <- methods_list[[method]]
  map_dfr(method_results, function(r) {
    tibble(
      method = method,
      n = r$n,
      r_median_us = r$r_median_us,
      rust_median_us = r$rust_median_us,
      speedup = r$speedup
    )
  })
})

# Filter to n=100,000 and prepare for plotting
speedup_data <- benchmark_df %>%
  filter(n == 100000, speedup > 0) %>%
  mutate(
    # Add category labels
    category = case_when(
      grepl("^ols", method, ignore.case = TRUE) ~ "Regression",
      grepl("^panel", method, ignore.case = TRUE) ~ "Panel",
      grepl("logit|probit", method, ignore.case = TRUE) ~ "Discrete",
      grepl("kmeans|pca|dbscan|hierarchical", method, ignore.case = TRUE) ~ "ML",
      grepl("arima|mstl|stl|holt|ar$", method, ignore.case = TRUE) ~ "Time Series",
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
    # Flag methods where Rust is faster
    faster = speedup >= 1,
    speedup_label = sprintf("%.1fx", speedup)
  ) %>%
  arrange(desc(speedup)) %>%
  mutate(method_label = factor(method_label, levels = rev(method_label)))

## PLOT ----
p <- ggplot(speedup_data, aes(x = method_label, y = speedup, fill = category)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_text(aes(label = speedup_label), hjust = -0.1, size = 2.8, color = "gray30") +
  coord_flip() +
  scale_y_continuous(
    name = "Speedup Factor (R time / Rust time)",
    limits = c(0, max(speedup_data$speedup) * 1.15),
    expand = c(0, 0)
  ) +
  scale_x_discrete(name = NULL) +
  scale_fill_brewer(palette = "Set2", name = "Category") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 10),
    plot.margin = margin(10, 20, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "benchmark_speedup.pdf"),
  plot = p,
  width = 8,
  height = 6
)

message("Created: ", OUTPUT, "benchmark_speedup.pdf")
