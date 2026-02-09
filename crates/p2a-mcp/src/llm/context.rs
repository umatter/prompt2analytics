//! Context management for LLM conversations.
//!
//! This module provides utilities for:
//! - Building rich dataset context for the system prompt
//! - Managing conversation context window limits
//! - Summarizing and pruning conversation history

use p2a_core::polars::prelude::*;
use p2a_core::Dataset;
use std::collections::HashMap;

/// Maximum number of sample values to include per column
const MAX_SAMPLE_VALUES: usize = 3;

/// Maximum length of a single sample value string
const MAX_SAMPLE_VALUE_LEN: usize = 50;

/// Build enhanced dataset context for the system prompt.
///
/// This provides much richer information than just column names:
/// - Data types for each column
/// - Sample values (first few non-null values)
/// - Basic statistics for numeric columns
/// - Unique value counts for categorical columns
pub fn build_enhanced_dataset_context(datasets: &HashMap<String, Dataset>) -> Option<String> {
    if datasets.is_empty() {
        return None;
    }

    let context_lines: Vec<String> = datasets
        .iter()
        .map(|(name, ds)| build_single_dataset_context(name, ds))
        .collect();

    Some(context_lines.join("\n\n"))
}

/// Build context for a single dataset with enhanced column information.
fn build_single_dataset_context(name: &str, ds: &Dataset) -> String {
    let df = ds.df();
    let mut lines = Vec::new();

    // Header with dimensions
    lines.push(format!(
        "### Dataset: `{}`\n- **Dimensions**: {} rows × {} columns",
        name,
        df.height(),
        df.width()
    ));

    // Column details
    lines.push("- **Columns**:".to_string());

    for col_name in df.get_column_names() {
        if let Ok(column) = df.column(col_name) {
            let col_info = build_column_info(col_name, column);
            lines.push(format!("  - {}", col_info));
        }
    }

    lines.join("\n")
}

/// Build detailed information for a single column.
fn build_column_info(col_name: &str, column: &Column) -> String {
    let series = column.as_materialized_series();
    let dtype = series.dtype();
    let null_count = series.null_count();
    let total_count = series.len();

    // Get dtype as a friendly string
    let dtype_str = match dtype {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => "integer",
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => "unsigned int",
        DataType::Float32 | DataType::Float64 => "float",
        DataType::Boolean => "boolean",
        DataType::String => "string",
        DataType::Date => "date",
        DataType::Datetime(_, _) => "datetime",
        DataType::Categorical(_, _) => "categorical",
        _ => "other",
    };

    // Get sample values
    let samples = get_sample_values(series, MAX_SAMPLE_VALUES);

    // Build the info string
    let mut parts = vec![format!("`{}` ({})", col_name, dtype_str)];

    // Add samples if available
    if !samples.is_empty() {
        parts.push(format!("e.g., {}", samples.join(", ")));
    }

    // Add null info if there are nulls
    if null_count > 0 {
        let null_pct = (null_count as f64 / total_count as f64 * 100.0).round();
        parts.push(format!("{}% null", null_pct));
    }

    // Add statistics for numeric columns
    if dtype.is_numeric() {
        if let Some(stats) = get_numeric_stats(series) {
            parts.push(stats);
        }
    } else if dtype == &DataType::String || matches!(dtype, DataType::Categorical(_, _)) {
        // For categorical/string columns, show unique count
        if let Ok(unique_count) = series.n_unique() {
            parts.push(format!("{} unique", unique_count));
        }
    }

    parts.join(" | ")
}

/// Get sample values from a series as strings.
fn get_sample_values(series: &Series, max_samples: usize) -> Vec<String> {
    let mut samples = Vec::new();
    let null_mask = series.is_null();

    for i in 0..series.len().min(max_samples * 2) {
        if samples.len() >= max_samples {
            break;
        }

        // Skip nulls
        if null_mask.get(i).unwrap_or(true) {
            continue;
        }

        // Get the value as a string
        let value_str = match series.get(i) {
            Ok(any_val) => {
                let s = format!("{}", any_val);
                // Truncate long values
                if s.len() > MAX_SAMPLE_VALUE_LEN {
                    format!("{}...", &s[..MAX_SAMPLE_VALUE_LEN])
                } else {
                    s
                }
            }
            Err(_) => continue,
        };

        // Skip empty strings
        if value_str.is_empty() || value_str == "null" {
            continue;
        }

        samples.push(format!("\"{}\"", value_str));
    }

    samples
}

/// Get basic statistics for a numeric series.
fn get_numeric_stats(series: &Series) -> Option<String> {
    // Try to cast to f64 for statistics
    let float_series = series.cast(&DataType::Float64).ok()?;
    let ca = float_series.f64().ok()?;

    let min = ca.min()?;
    let max = ca.max()?;

    // Format nicely - use integer format if values are whole numbers
    let format_num = |n: f64| {
        if n.fract() == 0.0 && n.abs() < 1e9 {
            format!("{}", n as i64)
        } else if n.abs() >= 1e6 || n.abs() < 0.001 {
            format!("{:.2e}", n)
        } else {
            format!("{:.2}", n)
        }
    };

    Some(format!("range: {}..{}", format_num(min), format_num(max)))
}

/// Configuration for context window management.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum total tokens for the context window
    pub max_tokens: usize,
    /// Tokens to reserve for the response
    pub reserved_for_response: usize,
    /// Number of recent turns to keep verbatim
    pub verbatim_turns: usize,
    /// Maximum length of a single tool result
    pub max_tool_result_length: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128_000, // Claude/GPT-4 default
            reserved_for_response: 4_096,
            verbatim_turns: 4,
            max_tool_result_length: 2000,
        }
    }
}

/// Truncate a tool result to a maximum length, preserving key information.
pub fn truncate_tool_result(result: &str, max_length: usize) -> String {
    if result.len() <= max_length {
        return result.to_string();
    }

    // Try to find a good truncation point
    let truncate_at = max_length.saturating_sub(50);

    // Look for a newline near the truncation point to avoid cutting mid-line
    let actual_truncate = result[..truncate_at]
        .rfind('\n')
        .unwrap_or(truncate_at);

    format!(
        "{}...\n\n[Result truncated: {} chars total, showing first {}]",
        &result[..actual_truncate],
        result.len(),
        actual_truncate
    )
}

/// Summarize tool calls for context compression.
///
/// Creates a compact summary of tool calls when full results would be too long.
pub fn summarize_tool_call(tool_name: &str, arguments: &serde_json::Value, result: &str) -> String {
    // Extract key info based on tool type
    let result_summary = if result.len() > 200 {
        // Try to extract the most important part
        let first_line = result.lines().next().unwrap_or("");
        if first_line.len() > 100 {
            format!("{}...", &first_line[..100])
        } else {
            first_line.to_string()
        }
    } else {
        result.to_string()
    };

    // Summarize arguments
    let args_summary = if let Some(obj) = arguments.as_object() {
        obj.iter()
            .filter(|(k, _)| *k != "csv_content") // Skip large inline data
            .map(|(k, v)| {
                let v_str = match v {
                    serde_json::Value::String(s) if s.len() > 20 => {
                        format!("\"{}...\"", &s[..20])
                    }
                    _ => v.to_string(),
                };
                format!("{}={}", k, v_str)
            })
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        String::new()
    };

    format!("{}({}) → {}", tool_name, args_summary, result_summary)
}

// =============================================================================
// Sliding Window Context Management
// =============================================================================

use super::provider::{Message, MessageRole, ToolCall as LlmToolCall, ToolResult as LlmToolResult};
use super::tokens::{estimate_message_tokens, estimate_tokens, ContextBudget};

/// Context manager for handling conversation history within token limits.
///
/// This implements a sliding window approach:
/// 1. Keep the most recent N turns verbatim
/// 2. Summarize older turns into a compact format
/// 3. Prune large tool results while preserving key information
pub struct ContextManager {
    config: ContextConfig,
    budget: ContextBudget,
}

impl ContextManager {
    /// Create a new context manager with the given configuration.
    pub fn new(config: ContextConfig) -> Self {
        Self {
            budget: ContextBudget::new(config.max_tokens)
                .with_response_reserve(config.reserved_for_response),
            config,
        }
    }

    /// Create a context manager for a specific model.
    pub fn for_model(model: &str) -> Self {
        let budget = ContextBudget::for_model(model);
        Self {
            config: ContextConfig {
                max_tokens: budget.max_tokens,
                ..Default::default()
            },
            budget,
        }
    }

    /// Process conversation history to fit within the context window.
    ///
    /// Returns a processed history with:
    /// - Recent turns kept verbatim
    /// - Older turns summarized
    /// - Large tool results truncated
    pub fn process_history(
        &mut self,
        messages: Vec<Message>,
        system_prompt: &str,
    ) -> ProcessedContext {
        // Update budget with system prompt
        self.budget.set_system_prompt(system_prompt);

        // Count turns (user + assistant pairs)
        let turns = count_turns(&messages);

        // If we have few enough turns, just truncate tool results
        if turns <= self.config.verbatim_turns {
            let processed = self.truncate_tool_results(messages);
            self.budget.set_history(&processed);

            return ProcessedContext {
                messages: processed,
                summary: None,
                tokens_used: self.budget.total_used(),
                turns_summarized: 0,
            };
        }

        // Split into old and recent turns
        let (old_messages, recent_messages) =
            split_at_turn(&messages, turns - self.config.verbatim_turns);

        // Summarize old messages
        let summary = self.summarize_messages(&old_messages);

        // Truncate tool results in recent messages
        let recent_processed = self.truncate_tool_results(recent_messages);

        // Build final history
        let mut final_messages = Vec::new();

        // Add summary as a system-style context message
        if !summary.is_empty() {
            final_messages.push(Message {
                role: MessageRole::User,
                content: format!(
                    "[Conversation context from earlier in this session:]\n{}",
                    summary
                ),
                tool_calls: None,
                tool_results: None,
            });
            // Add a brief acknowledgment to maintain turn structure
            final_messages.push(Message {
                role: MessageRole::Assistant,
                content: "Understood, I'll keep this context in mind.".to_string(),
                tool_calls: None,
                tool_results: None,
            });
        }

        final_messages.extend(recent_processed);

        self.budget.set_history(&final_messages);

        ProcessedContext {
            messages: final_messages,
            summary: Some(summary),
            tokens_used: self.budget.total_used(),
            turns_summarized: turns - self.config.verbatim_turns,
        }
    }

    /// Truncate tool results in messages to fit within limits.
    fn truncate_tool_results(&self, messages: Vec<Message>) -> Vec<Message> {
        messages
            .into_iter()
            .map(|mut msg| {
                // Truncate tool results
                if let Some(ref mut results) = msg.tool_results {
                    for result in results.iter_mut() {
                        if result.content.len() > self.config.max_tool_result_length {
                            result.content = truncate_tool_result(
                                &result.content,
                                self.config.max_tool_result_length,
                            );
                        }
                    }
                }
                msg
            })
            .collect()
    }

    /// Summarize a sequence of messages into a compact format.
    fn summarize_messages(&self, messages: &[Message]) -> String {
        let mut summaries = Vec::new();
        let mut current_user_msg: Option<&str> = None;

        for msg in messages {
            match msg.role {
                MessageRole::User => {
                    // Save user message for context
                    current_user_msg = Some(&msg.content);
                }
                MessageRole::Assistant => {
                    // Summarize what the assistant did
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            let summary =
                                summarize_tool_call(&tc.name, &tc.arguments, "[result omitted]");
                            summaries.push(format!("• {}", summary));
                        }
                    } else if !msg.content.is_empty() {
                        // Just text response - summarize it
                        let preview = if msg.content.len() > 100 {
                            format!("{}...", &msg.content[..100])
                        } else {
                            msg.content.clone()
                        };
                        if let Some(user_q) = current_user_msg {
                            let user_preview = if user_q.len() > 50 {
                                format!("{}...", &user_q[..50])
                            } else {
                                user_q.to_string()
                            };
                            summaries.push(format!("• Q: \"{}\" → A: \"{}\"", user_preview, preview));
                        } else {
                            summaries.push(format!("• Response: \"{}\"", preview));
                        }
                    }
                    current_user_msg = None;
                }
                MessageRole::Tool => {
                    // Tool results - extract key info
                    if let Some(results) = &msg.tool_results {
                        for result in results {
                            if result.is_error {
                                summaries.push(format!("  ⚠ Error: {}", &result.content[..result.content.len().min(50)]));
                            }
                            // Success results are already summarized above
                        }
                    }
                }
                MessageRole::System => {
                    // Skip system messages in summary
                }
            }
        }

        summaries.join("\n")
    }

    /// Get the current budget status.
    pub fn budget(&self) -> &ContextBudget {
        &self.budget
    }

    /// Check if more history can be added within budget.
    pub fn can_add_history(&self, additional_tokens: usize) -> bool {
        self.budget.remaining() >= additional_tokens
    }
}

/// Result of processing conversation context.
#[derive(Debug)]
pub struct ProcessedContext {
    /// The processed messages ready to send to the LLM
    pub messages: Vec<Message>,
    /// Summary of older messages (if any were summarized)
    pub summary: Option<String>,
    /// Total tokens used
    pub tokens_used: usize,
    /// Number of turns that were summarized
    pub turns_summarized: usize,
}

/// Count the number of turns (user+assistant pairs) in a message list.
fn count_turns(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .count()
}

/// Split messages at a specific turn boundary.
fn split_at_turn(messages: &[Message], turn: usize) -> (Vec<Message>, Vec<Message>) {
    let mut user_count = 0;
    let mut split_index = 0;

    for (i, msg) in messages.iter().enumerate() {
        if msg.role == MessageRole::User {
            user_count += 1;
            if user_count > turn {
                split_index = i;
                break;
            }
        }
    }

    if split_index == 0 {
        (Vec::new(), messages.to_vec())
    } else {
        (
            messages[..split_index].to_vec(),
            messages[split_index..].to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_tool_result_short() {
        let result = "Short result";
        assert_eq!(truncate_tool_result(result, 100), result);
    }

    #[test]
    fn test_truncate_tool_result_long() {
        let result = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6";
        let truncated = truncate_tool_result(result, 20);
        assert!(truncated.contains("truncated"));
        assert!(truncated.len() < result.len() + 100); // Some overhead for message
    }

    #[test]
    fn test_summarize_tool_call() {
        let args = serde_json::json!({"dataset": "sales", "y": "price"});
        let result = "Successfully ran regression";
        let summary = summarize_tool_call("regression_ols", &args, result);
        assert!(summary.contains("regression_ols"));
        assert!(summary.contains("dataset="));
    }

    #[test]
    fn test_count_turns() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
            Message::user("How are you?"),
            Message::assistant("I'm good!"),
        ];
        assert_eq!(count_turns(&messages), 2);
    }

    #[test]
    fn test_split_at_turn() {
        let messages = vec![
            Message::user("Turn 1"),
            Message::assistant("Response 1"),
            Message::user("Turn 2"),
            Message::assistant("Response 2"),
            Message::user("Turn 3"),
            Message::assistant("Response 3"),
        ];

        let (old, recent) = split_at_turn(&messages, 1);
        assert_eq!(old.len(), 2); // Turn 1
        assert_eq!(recent.len(), 4); // Turns 2 and 3
    }

    #[test]
    fn test_context_manager_few_turns() {
        let mut manager = ContextManager::new(ContextConfig::default());
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi!"),
        ];

        let result = manager.process_history(messages, "You are helpful.");
        assert!(result.summary.is_none());
        assert_eq!(result.turns_summarized, 0);
        assert_eq!(result.messages.len(), 2);
    }

    #[test]
    fn test_context_manager_many_turns() {
        let config = ContextConfig {
            verbatim_turns: 2,
            ..Default::default()
        };
        let mut manager = ContextManager::new(config);

        let messages = vec![
            Message::user("Turn 1"),
            Message::assistant("Response 1"),
            Message::user("Turn 2"),
            Message::assistant("Response 2"),
            Message::user("Turn 3"),
            Message::assistant("Response 3"),
            Message::user("Turn 4"),
            Message::assistant("Response 4"),
        ];

        let result = manager.process_history(messages, "You are helpful.");
        assert!(result.summary.is_some());
        assert_eq!(result.turns_summarized, 2);
        // 2 for summary context + 4 for recent turns
        assert!(result.messages.len() >= 4);
    }
}
