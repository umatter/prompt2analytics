# ============================================================================
# fig_latency_comparison.R
# Box plot of latency distribution by model
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

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
read_jsonl <- function(file) {
  lines <- readLines(paste0(INPUT, file), warn = FALSE)
  map_dfr(lines, ~ fromJSON(.x))
}

all_results <- map_dfr(EVAL_FILES, read_jsonl)

# Clean model names and add provider info
latency_data <- all_results %>%
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
    labels = scales::comma
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
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "top",
    legend.justification = "left",
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
