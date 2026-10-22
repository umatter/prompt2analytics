//! Token counting and context budget management.
//!
//! This module provides utilities for estimating token counts and managing
//! context window budgets for LLM conversations.
//!
//! Token counting uses a simple approximation based on common observations:
//! - Average ~4 characters per token for English text
//! - Average ~0.75 words per token
//! - JSON structure adds overhead
//!
//! For more accurate counts with specific models, consider using tiktoken-rs.

use super::{Message, MessageRole, ToolDefinition};

/// Average characters per token (based on GPT-3/4 tokenizer observations)
const CHARS_PER_TOKEN: f64 = 4.0;

/// Overhead multiplier for JSON structure
const JSON_OVERHEAD: f64 = 1.15;

/// Estimate token count for a string.
pub fn estimate_tokens(text: &str) -> usize {
    estimate_tokens_for_chars(text.chars().count())
}

/// The token estimate of a text of `chars` characters (the measure of
/// [`estimate_tokens`], for callers that sum lengths without building the
/// text).
pub(crate) fn estimate_tokens_for_chars(chars: usize) -> usize {
    (chars as f64 / CHARS_PER_TOKEN).ceil() as usize
}

/// Estimate token count for a JSON value.
pub fn estimate_json_tokens(value: &serde_json::Value) -> usize {
    let json_str = serde_json::to_string(value).unwrap_or_default();
    let base_tokens = estimate_tokens(&json_str);
    // Add overhead for JSON structure
    (base_tokens as f64 * JSON_OVERHEAD).ceil() as usize
}

/// Estimate token count for a message.
pub fn estimate_message_tokens(message: &Message) -> usize {
    let mut total = 0;

    // Role overhead (approximately 4 tokens for the message structure)
    total += 4;

    // Content
    total += estimate_tokens(&message.content);

    // Tool calls
    if let Some(tool_calls) = &message.tool_calls {
        for tc in tool_calls {
            total += estimate_tokens(&tc.name);
            total += estimate_tokens(&tc.id);
            total += estimate_json_tokens(&tc.arguments);
            total += 10; // Overhead for tool call structure
        }
    }

    // Tool results
    if let Some(tool_results) = &message.tool_results {
        for tr in tool_results {
            total += estimate_tokens(&tr.tool_call_id);
            total += estimate_tokens(&tr.content);
            total += 8; // Overhead for tool result structure
        }
    }

    total
}

/// Estimate token count for a tool definition.
pub fn estimate_tool_definition_tokens(tool: &ToolDefinition) -> usize {
    let mut total = 0;
    total += estimate_tokens(&tool.name);
    total += estimate_tokens(&tool.description);
    total += estimate_json_tokens(&tool.parameters);
    total += 15; // Overhead for function schema structure
    total
}

/// Context budget for managing LLM context windows.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Maximum total tokens for the model's context window
    pub max_tokens: usize,
    /// Tokens reserved for the model's response
    pub reserved_for_response: usize,
    /// Current token count for the system prompt
    pub system_prompt_tokens: usize,
    /// Current token count for conversation history
    pub history_tokens: usize,
    /// Current token count for tool definitions
    pub tools_tokens: usize,
}

impl ContextBudget {
    /// Create a new context budget with default settings.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            reserved_for_response: 4096,
            system_prompt_tokens: 0,
            history_tokens: 0,
            tools_tokens: 0,
        }
    }

    /// Create a context budget for a specific model.
    pub fn for_model(model: &str) -> Self {
        let max_tokens = get_model_context_size(model);
        Self::new(max_tokens)
    }

    /// Set the reserved tokens for response.
    pub fn with_response_reserve(mut self, tokens: usize) -> Self {
        self.reserved_for_response = tokens;
        self
    }

    /// Calculate tokens used by the system prompt.
    pub fn set_system_prompt(&mut self, prompt: &str) {
        self.system_prompt_tokens = estimate_tokens(prompt) + 4; // +4 for message structure
    }

    /// Calculate tokens used by tool definitions.
    pub fn set_tools(&mut self, tools: &[ToolDefinition]) {
        self.tools_tokens = tools.iter().map(estimate_tool_definition_tokens).sum();
    }

    /// Calculate tokens used by conversation history.
    pub fn set_history(&mut self, messages: &[Message]) {
        self.history_tokens = messages.iter().map(estimate_message_tokens).sum();
    }

    /// Get total tokens currently used.
    pub fn total_used(&self) -> usize {
        self.system_prompt_tokens + self.history_tokens + self.tools_tokens
    }

    /// Get available tokens for history (after system prompt, tools, and response reserve).
    pub fn available_for_history(&self) -> usize {
        let overhead = self.system_prompt_tokens + self.tools_tokens + self.reserved_for_response;
        self.max_tokens.saturating_sub(overhead)
    }

    /// Get remaining tokens available (total available minus used).
    pub fn remaining(&self) -> usize {
        let total_available = self.max_tokens.saturating_sub(self.reserved_for_response);
        total_available.saturating_sub(self.total_used())
    }

    /// Check if the context is over budget.
    pub fn is_over_budget(&self) -> bool {
        self.total_used() + self.reserved_for_response > self.max_tokens
    }

    /// Calculate how many tokens need to be trimmed to fit within budget.
    pub fn tokens_over_budget(&self) -> usize {
        let total_needed = self.total_used() + self.reserved_for_response;
        total_needed.saturating_sub(self.max_tokens)
    }

    /// Get a summary of the budget allocation.
    pub fn summary(&self) -> String {
        format!(
            "Context Budget: {}/{} tokens used\n  System: {}\n  History: {}\n  Tools: {}\n  Reserved: {}\n  Available: {}",
            self.total_used(),
            self.max_tokens,
            self.system_prompt_tokens,
            self.history_tokens,
            self.tools_tokens,
            self.reserved_for_response,
            self.remaining()
        )
    }
}

/// The input context window of a hosted model whose window is known, or
/// `None`.
///
/// Any `provider/` prefix (OpenRouter style) is ignored, and case does not
/// matter. GPT-5 is its input budget (400k total minus 128k output).
pub fn known_model_context_size(model: &str) -> Option<usize> {
    let lower = model.to_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    let o_series = |p: &str| name == p || name.starts_with(&format!("{p}-"));

    if name.contains("claude-2") || name.contains("claude-instant") {
        Some(100_000)
    } else if name.contains("claude") {
        Some(200_000)
    } else if name.contains("gpt-5") {
        Some(272_000)
    } else if name.contains("gpt-4.1") {
        Some(1_047_576)
    } else if name.contains("gpt-4o") || name.contains("chatgpt-4o") {
        Some(128_000)
    } else if name.contains("gpt-4-turbo") || name.contains("gpt-4-1106") {
        Some(128_000)
    } else if name.contains("gpt-4-32k") {
        Some(32_768)
    } else if name.contains("gpt-4") {
        Some(8_192)
    } else if name.contains("gpt-3.5-turbo-16k") || name.contains("gpt-3.5") {
        Some(16_385)
    } else if name == "o1-mini" || name.starts_with("o1-preview") {
        Some(128_000)
    } else if o_series("o1") || o_series("o3") || o_series("o4") {
        Some(200_000)
    } else {
        None
    }
}

/// Get the context window size for a model: a known hosted model's window,
/// a conservative size for common local models, else 8,192.
pub fn get_model_context_size(model: &str) -> usize {
    if let Some(n) = known_model_context_size(model) {
        return n;
    }
    let model_lower = model.to_lowercase();

    // Ollama/local models (conservative default)
    if model_lower.contains("llama") {
        if model_lower.contains("70b") || model_lower.contains("405b") {
            return 8_192;
        }
        return 4_096;
    }
    if model_lower.contains("mistral") {
        return 32_768;
    }
    if model_lower.contains("mixtral") {
        return 32_768;
    }

    // Default fallback
    8_192
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        // Empty string
        assert_eq!(estimate_tokens(""), 0);

        // Short string (should be at least 1 token)
        assert!(estimate_tokens("hi") >= 1);

        // Longer text
        let text = "This is a longer piece of text that should result in multiple tokens.";
        let tokens = estimate_tokens(text);
        assert!(tokens > 10);
        assert!(tokens < 30);
    }

    #[test]
    fn test_context_budget() {
        let mut budget = ContextBudget::new(8192);
        budget.set_system_prompt("You are a helpful assistant.");

        assert!(budget.system_prompt_tokens > 0);
        assert!(!budget.is_over_budget());
        assert!(budget.remaining() > 0);
    }

    #[test]
    fn test_model_context_sizes() {
        for (m, n) in [
            ("claude-sonnet-4-20250514", 200_000),
            ("claude-opus-4-5-20251101", 200_000),
            ("claude-3-5-haiku-20241022", 200_000),
            ("anthropic/claude-3.5-sonnet", 200_000),
            ("claude-2.1", 100_000),
            ("gpt-4o-mini", 128_000),
            ("openai/gpt-4o-mini", 128_000),
            ("gpt-4.1-mini", 1_047_576),
            ("gpt-5", 272_000),
            ("gpt-5-mini", 272_000),
            ("o3-mini", 200_000),
            ("o4-mini", 200_000),
            ("o1", 200_000),
            ("o1-mini", 128_000),
            ("gpt-4-turbo", 128_000),
            ("gpt-4", 8_192),
            ("gpt-3.5-turbo", 16_385),
            ("unknown-model", 8_192),
        ] {
            assert_eq!(get_model_context_size(m), n, "{m}");
        }
        assert_eq!(known_model_context_size("llama3.2"), None);
        assert_eq!(known_model_context_size("unknown-model"), None);
        assert_eq!(
            known_model_context_size("claude-sonnet-4-20250514"),
            Some(200_000)
        );
    }
}
