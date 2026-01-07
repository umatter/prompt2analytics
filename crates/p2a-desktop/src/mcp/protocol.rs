//! MCP/JSON-RPC protocol message types.

use serde::{Deserialize, Serialize};

/// JSON-RPC 2.0 request.
#[derive(Debug, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: &str, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

/// MCP tool call parameters.
#[derive(Debug, Serialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// MCP tool call result.
#[derive(Debug, Deserialize)]
pub struct ToolCallResult {
    pub content: Vec<ContentItem>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

/// Content item in MCP response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentItem {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource {
        uri: String,
        #[serde(default)]
        mime_type: Option<String>,
    },
}

/// Parsed tool result for frontend consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub content: String,
    pub images: Vec<ImageData>,
    pub error: Option<String>,
}

/// Image data extracted from tool response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub base64: String,
    pub alt: String,
}

impl ToolResult {
    /// Parse a ToolCallResult into a frontend-friendly ToolResult.
    pub fn from_call_result(result: ToolCallResult, tool_name: &str) -> Self {
        let mut text_parts = Vec::new();
        let mut images = Vec::new();

        for item in result.content {
            match item {
                ContentItem::Text { text } => {
                    // Check for embedded base64 image
                    if let Some((before, base64)) = extract_base64_image(&text) {
                        if !before.trim().is_empty() {
                            text_parts.push(before);
                        }
                        images.push(ImageData {
                            base64,
                            alt: tool_name.to_string(),
                        });
                    } else {
                        text_parts.push(text);
                    }
                }
                ContentItem::Image { data, .. } => {
                    images.push(ImageData {
                        base64: data,
                        alt: tool_name.to_string(),
                    });
                }
                ContentItem::Resource { .. } => {
                    // Resources not currently handled
                }
            }
        }

        let content = text_parts.join("\n");
        ToolResult {
            success: !result.is_error,
            content: content.clone(),
            images,
            error: if result.is_error {
                Some(content)
            } else {
                None
            },
        }
    }
}

/// Extract base64 image data from text output.
///
/// Visualization tools embed images like:
/// ```text
/// Image (base64 PNG, 12345 bytes):
/// iVBORw0KGgoAAAANS...
/// ```
fn extract_base64_image(text: &str) -> Option<(String, String)> {
    // Look for the pattern
    if let Some(idx) = text.find("Image (base64") {
        let before = text[..idx].to_string();
        // Find the newline after the header
        if let Some(newline_idx) = text[idx..].find('\n') {
            let start = idx + newline_idx + 1;
            let base64 = text[start..].trim().to_string();
            if !base64.is_empty() {
                return Some((before, base64));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_base64_image() {
        let text = "Some text\n\nImage (base64 PNG, 100 bytes):\niVBORw0KGgo=";
        let result = extract_base64_image(text);
        assert!(result.is_some());
        let (before, base64) = result.unwrap();
        assert_eq!(before, "Some text\n\n");
        assert_eq!(base64, "iVBORw0KGgo=");
    }

    #[test]
    fn test_no_image() {
        let text = "Just some text without an image";
        assert!(extract_base64_image(text).is_none());
    }
}
