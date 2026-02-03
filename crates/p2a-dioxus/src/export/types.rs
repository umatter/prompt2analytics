//! Export types and data structures

use serde::{Deserialize, Serialize};

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Full JSON archive (can be re-imported)
    Json,
    /// Human-readable Markdown
    Markdown,
    /// Self-contained HTML report
    Html,
}

impl ExportFormat {
    /// Get the file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Markdown => "md",
            ExportFormat::Html => "html",
        }
    }

    /// Get the MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::Json => "application/json",
            ExportFormat::Markdown => "text/markdown",
            ExportFormat::Html => "text/html",
        }
    }
}

/// A tool call in the exported format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
}

/// A message in the exported format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ExportedToolCall>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub images: Vec<String>,
    pub timestamp: String,
}

/// Conversation metadata in the exported format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedConversationMeta {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// A dataset reference used in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedDataset {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cols: Option<u32>,
}

/// Complete exported conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedConversation {
    /// Export format version for future compatibility
    pub version: String,
    /// Export timestamp
    pub exported_at: String,
    /// Conversation metadata
    pub conversation: ExportedConversationMeta,
    /// All messages in the conversation
    pub messages: Vec<ExportedMessage>,
    /// Datasets referenced in tool calls
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub datasets: Vec<ExportedDataset>,
}

/// Options for JSON export
#[derive(Debug, Clone, Default)]
pub struct JsonExportOptions {
    /// Pretty-print the JSON output
    pub pretty: bool,
    /// Include tool call results (can be large)
    pub include_tool_results: bool,
    /// Include base64 images (can be very large)
    pub include_images: bool,
}

/// Options for Markdown export
#[derive(Debug, Clone)]
pub struct MarkdownExportOptions {
    /// Include tool calls in collapsible sections
    pub include_tool_calls: bool,
    /// Include images as base64 data URIs
    pub include_images: bool,
    /// Include timestamps for each message
    pub include_timestamps: bool,
}

impl Default for MarkdownExportOptions {
    fn default() -> Self {
        Self {
            include_tool_calls: true,
            include_images: true,
            include_timestamps: false,
        }
    }
}

/// Options for HTML export
#[derive(Debug, Clone)]
pub struct HtmlExportOptions {
    /// Use dark theme (vs light)
    pub dark_theme: bool,
    /// Include tool calls in collapsible sections
    pub include_tool_calls: bool,
    /// Include images inline
    pub include_images: bool,
}

impl Default for HtmlExportOptions {
    fn default() -> Self {
        Self {
            dark_theme: false,
            include_tool_calls: true,
            include_images: true,
        }
    }
}
