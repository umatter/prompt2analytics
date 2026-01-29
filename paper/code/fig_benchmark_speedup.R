# ============================================================================
# fig_benchmark_speedup.R
# Horizontal bar chart showing Rust/R speedup factors at n=100,000
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
source("helpers.R")

## DATA IMPORT AND PREPARATION ----
benchmark_df <- load_benchmark_summary(paste0(INPUT_BENCHMARK, "benchmark_summary.json"))

# Filter to n=100,000 and prepare for plotting
speedup_data <- benchmark_df %>%
  filter(n == 100000, speedup > 0) %>%
  mutate(
    category = categorize_method(method),
    method_label = clean_method_label(method),
    faster = speedup >= 1,
    speedup_label = format_speedup(speedup)
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
  paste0(OUTPUT_FIGURES, "benchmark_speedup.pdf"),
  plot = p,
  width = 8,
  height = 6
)

message("Created: ", OUTPUT_FIGURES, "benchmark_speedup.pdf")
