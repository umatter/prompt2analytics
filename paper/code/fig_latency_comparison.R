# ============================================================================
# fig_latency_comparison.R
# Box plot of latency distribution by model
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

INPUT <- "llm_eval/results/naturalistic/"
OUTPUT <- "../figures/"

## DATA IMPORT AND PREPARATION ----

# Read concatenated pretty-printed JSON objects from a file
read_json_concat <- function(file) {
  text <- readLines(file, warn = FALSE)
  # Split on lines that are just "}" followed by "{"
  json_str <- paste(text, collapse = "\n")
  # Wrap in array: insert commas between top-level objects
  json_str <- gsub("\\}\n\\{", "},{", json_str)
  json_str <- paste0("[", json_str, "]")
  fromJSON(json_str, flatten = TRUE)
}

# Read all files matching the 7 models used in the paper
eval_files <- list.files(INPUT, pattern = "_all_.*\\.jsonl$", full.names = TRUE)

# Filter to the 7 models in the paper
keep_models <- c("claude-3-5-haiku", "qwen-2.5-72b", "llama-3.3-70b",
                 "gpt-4o-mini", "gpt-5-nano", "gpt-4.1-nano", "ministral-3b")
eval_files <- eval_files[sapply(eval_files, function(f) {
  any(sapply(keep_models, function(m) grepl(m, f)))
})]

all_results <- map_dfr(eval_files, read_json_concat)

# Clean model names and add provider info
latency_data <- all_results %>%
  filter(!is.na(latency_ms)) %>%
  mutate(
    model_display = case_when(
      grepl("claude", model) ~ "Claude 3.5\nHaiku",
      grepl("qwen-2.5-72b", model) ~ "Qwen 2.5\n72B",
      grepl("llama-3.3-70b", model) ~ "Llama 3.3\n70B",
      model == "gpt-4o-mini" ~ "GPT-4o\nMini",
      grepl("gpt-5-nano", model) ~ "GPT-5\nNano",
      grepl("gpt-4.1-nano", model) ~ "GPT-4.1\nNano",
      grepl("ministral-3b", model) ~ "Ministral\n3B",
      TRUE ~ model
    ),
    provider = case_when(
      grepl("claude", model) ~ "Anthropic",
      grepl("gpt", model) ~ "OpenAI",
      TRUE ~ "Open Source"
    )
  )

# Calculate median latency for ordering
model_order <- latency_data %>%
  group_by(model_display) %>%
  summarise(median_latency = median(latency_ms), .groups = "drop") %>%
  arrange(median_latency) %>%
  pull(model_display)

latency_data <- latency_data %>%
  mutate(model_display = factor(model_display, levels = model_order))

## PLOT ----
p <- ggplot(latency_data, aes(x = model_display, y = latency_ms, fill = provider)) +
  geom_boxplot(outlier.size = 1, outlier.alpha = 0.5) +
  scale_y_log10(
    breaks = c(500, 1000, 2000, 5000, 10000, 20000),
    labels = scales::comma,
    limits = c(300, 30000)
  ) +
  scale_fill_manual(
    values = c(
      "OpenAI" = "#74aa9c",
      "Anthropic" = "#d4a27f",
      "Open Source" = "#7facd4"
    ),
    name = "Provider"
  ) +
  labs(
    x = NULL,
    y = "Latency (ms, log scale)",
    title = NULL
  ) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.justification = "center",
    plot.margin = margin(10, 20, 10, 10)
  ) +
  annotation_logticks(sides = "l", size = 0.3)

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "fig_latency_comparison.pdf"),
  plot = p,
  width = 7,
  height = 5,
  device = cairo_pdf
)

message("Created: ", OUTPUT, "fig_latency_comparison.pdf")
