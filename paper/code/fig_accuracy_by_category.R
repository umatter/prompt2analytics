# ============================================================================
# fig_accuracy_by_category.R
# Heatmap showing accuracy by category for each model
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
library(viridis)

INPUT <- "llm_eval/results/"
OUTPUT <- "../figures/"

# Complete evaluation files (87 tests)
EVAL_FILES <- c(
  "claude-3-5-haiku-20241022_all_20260123_165810.jsonl",
  "qwen_qwen-2.5-72b-instruct_all_20260123_173345.jsonl",
  "meta-llama_llama-3.3-70b-instruct_all_20260123_171743.jsonl",
  "gpt-4o-mini_all_20260123_152151.jsonl",
  "gpt-4.1-nano-2025-04-14_all_20260123_155259.jsonl",
  "gpt-5-nano-2025-08-07_all_20260123_161312.jsonl",
  "mistralai_ministral-3b_all_20260123_205903.jsonl"
)

## DATA IMPORT AND PREPARATION ----
# Read all JSONL files and aggregate by category
read_jsonl <- function(file) {
  lines <- readLines(paste0(INPUT, file), warn = FALSE)
  map_dfr(lines, ~ fromJSON(.x))
}

all_results <- map_dfr(EVAL_FILES, read_jsonl)

# Calculate accuracy by model and category
# Note: Only "exact" and "acceptable" matches count toward accuracy
# "category" matches (correct category, wrong specific tool) do not count
category_accuracy <- all_results %>%
  group_by(model, category) %>%
  summarise(
    correct = sum(match_type %in% c("exact", "acceptable")),
    total = n(),
    accuracy = correct / total * 100,
    .groups = "drop"
  ) %>%
  mutate(
    # Clean model names
    model_display = case_when(
      grepl("claude", model) ~ "Claude 3.5 Haiku",
      grepl("qwen-2.5-72b", model) ~ "Qwen 2.5 72B",
      grepl("llama-3.3-70b", model) ~ "Llama 3.3 70B",
      model == "gpt-4o-mini" ~ "GPT-4o Mini",
      grepl("gpt-5-nano", model) ~ "GPT-5 Nano",
      grepl("gpt-4.1-nano", model) ~ "GPT-4.1 Nano",
      grepl("ministral-3b", model) ~ "Ministral 3B",
      TRUE ~ model
    ),
    # Clean category names
    category_display = case_when(
      category == "regression" ~ "Regression",
      category == "panel" ~ "Panel",
      category == "causal" ~ "Causal",
      category == "discrete" ~ "Discrete",
      category == "timeseries" ~ "Time Series",
      category == "hypothesis" ~ "Hypothesis",
      category == "ml" ~ "ML",
      category == "viz" ~ "Visualization",
      TRUE ~ category
    )
  )

# Calculate overall accuracy for ordering
model_order <- category_accuracy %>%
  group_by(model_display) %>%
  summarise(overall = mean(accuracy), .groups = "drop") %>%
  arrange(desc(overall)) %>%
  pull(model_display)

# Set factor levels
category_order <- c("Regression", "Panel", "Causal", "Discrete",
                    "Time Series", "Hypothesis", "ML", "Visualization")

category_accuracy <- category_accuracy %>%
  mutate(
    model_display = factor(model_display, levels = rev(model_order)),
    category_display = factor(category_display, levels = category_order)
  )

## PLOT ----
p <- ggplot(category_accuracy, aes(x = category_display, y = model_display, fill = accuracy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", accuracy)),
            size = 3.5, color = ifelse(category_accuracy$accuracy < 70, "white", "black")) +
  scale_fill_viridis(
    option = "D",
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    labels = function(x) paste0(x, "%"),
    name = "Accuracy"
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.margin = margin(10, 10, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "fig_accuracy_by_category.pdf"),
  plot = p,
  width = 8,
  height = 5,
  device = cairo_pdf
)

message("Created: ", OUTPUT, "fig_accuracy_by_category.pdf")
