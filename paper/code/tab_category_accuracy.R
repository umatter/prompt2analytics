# ============================================================================
# tab_category_accuracy.R
# Per-category accuracy breakdown (LaTeX)
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
library(xtable)

INPUT <- "llm_eval/results/"
OUTPUT <- "../tables/"

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
    model_short = case_when(
      grepl("claude", model) ~ "Haiku",
      grepl("qwen-2.5-72b", model) ~ "Qwen",
      grepl("llama-3.3-70b", model) ~ "Llama",
      model == "gpt-4o-mini" ~ "4o-mini",
      grepl("gpt-5-nano", model) ~ "5-nano",
      grepl("gpt-4.1-nano", model) ~ "4.1-nano",
      grepl("ministral-3b", model) ~ "Mini-3B",
      TRUE ~ model
    )
  )

# Calculate overall accuracy per model
overall_accuracy <- all_results %>%
  group_by(model) %>%
  summarise(accuracy = sum(match_type %in% c("exact", "acceptable")) / n() * 100, .groups = "drop") %>%
  mutate(
    model_short = case_when(
      grepl("claude", model) ~ "Haiku",
      grepl("qwen-2.5-72b", model) ~ "Qwen",
      grepl("llama-3.3-70b", model) ~ "Llama",
      model == "gpt-4o-mini" ~ "4o-mini",
      grepl("gpt-5-nano", model) ~ "5-nano",
      grepl("gpt-4.1-nano", model) ~ "4.1-nano",
      grepl("ministral-3b", model) ~ "Mini-3B",
      TRUE ~ model
    ),
    category = "Overall"
  )

# Pivot to wide format for table
model_order <- overall_accuracy %>%
  arrange(desc(accuracy)) %>%
  pull(model_short)

category_order <- c("regression", "panel", "causal", "discrete",
                    "timeseries", "hypothesis", "ml", "viz")

table_data <- category_accuracy %>%
  select(category, model_short, accuracy) %>%
  pivot_wider(names_from = model_short, values_from = accuracy) %>%
  mutate(
    Category = case_when(
      category == "regression" ~ "Regression",
      category == "panel" ~ "Panel Data",
      category == "causal" ~ "Causal Inference",
      category == "discrete" ~ "Discrete Choice",
      category == "timeseries" ~ "Time Series",
      category == "hypothesis" ~ "Hypothesis Testing",
      category == "ml" ~ "Machine Learning",
      category == "viz" ~ "Visualization",
      TRUE ~ category
    )
  ) %>%
  arrange(match(category, category_order)) %>%
  select(Category, all_of(model_order))

# Add overall row
overall_row <- overall_accuracy %>%
  select(model_short, accuracy) %>%
  pivot_wider(names_from = model_short, values_from = accuracy) %>%
  mutate(Category = "\\textbf{Overall}") %>%
  select(Category, all_of(model_order))

table_data <- bind_rows(table_data, overall_row)

# Format percentages
table_data <- table_data %>%
  mutate(across(all_of(model_order), ~ sprintf("%.1f", .x)))

## CREATE LATEX TABLE ----
latex_table <- xtable(
  table_data,
  caption = "Accuracy by test category (\\%). Each category contains 8--13 test cases covering basic to advanced use cases. Bold row shows overall accuracy across all 87 test cases.",
  label = "tab:category-accuracy",
  align = c("l", "l", rep("r", length(model_order)))
)

## WRITE TO DISK ----
print(
  latex_table,
  file = paste0(OUTPUT, "tab_category_accuracy.tex"),
  include.rownames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = identity,
  caption.placement = "top",
  table.placement = "htbp",
  floating = TRUE,
  hline.after = c(-1, 0, nrow(table_data) - 1, nrow(table_data))
)

message("Created: ", OUTPUT, "tab_category_accuracy.tex")
