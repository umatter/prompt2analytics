//! JSON export functionality

use chrono::Utc;

use crate::api::Conversation;
use crate::state::ChatMessage;

use super::types::{
    ExportedConversation, ExportedConversationMeta, ExportedDataset, ExportedMessage,
    ExportedToolCall, JsonExportOptions,
};

/// Export a conversation and its messages to JSON format
pub fn export_to_json(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &JsonExportOptions,
) -> Result<String, String> {
    let exported = build_exported_conversation(conversation, messages, options);

    if options.pretty {
        serde_json::to_string_pretty(&exported)
            .map_err(|e| format!("JSON serialization error: {}", e))
    } else {
        serde_json::to_string(&exported).map_err(|e| format!("JSON serialization error: {}", e))
    }
}

/// Build the exported conversation structure
fn build_exported_conversation(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &JsonExportOptions,
) -> ExportedConversation {
    let now = Utc::now();

    // Build conversation metadata
    let conversation_meta = if let Some(conv) = conversation {
        ExportedConversationMeta {
            id: conv.id.clone(),
            title: conv.title.clone(),
            created_at: conv.created_at.clone(),
            updated_at: conv.updated_at.clone(),
            session_id: Some(conv.session_id.clone()),
        }
    } else {
        ExportedConversationMeta {
            id: uuid::Uuid::new_v4().to_string(),
            title: "Untitled Conversation".to_string(),
            created_at: now.to_rfc3339(),
            updated_at: now.to_rfc3339(),
            session_id: None,
        }
    };

    // Convert messages
    let exported_messages: Vec<ExportedMessage> = messages
        .iter()
        .filter(|m| !m.is_streaming) // Don't export incomplete streaming messages
        .map(|m| convert_message(m, options))
        .collect();

    // Extract dataset references from tool call arguments
    let datasets = extract_datasets(messages);

    ExportedConversation {
        version: "1.0".to_string(),
        exported_at: now.to_rfc3339(),
        conversation: conversation_meta,
        messages: exported_messages,
        datasets,
    }
}

/// Convert a ChatMessage to ExportedMessage
fn convert_message(msg: &ChatMessage, options: &JsonExportOptions) -> ExportedMessage {
    let tool_calls: Vec<ExportedToolCall> = msg
        .tool_calls
        .iter()
        .map(|tc| ExportedToolCall {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            result: if options.include_tool_results {
                tc.result.clone()
            } else {
                None
            },
            success: tc.success,
        })
        .collect();

    let images = if options.include_images {
        msg.images.clone()
    } else {
        Vec::new()
    };

    ExportedMessage {
        id: msg.id.clone(),
        role: msg.role.clone(),
        content: msg.content.clone(),
        tool_calls,
        images,
        timestamp: msg.timestamp.to_rfc3339(),
    }
}

/// Extract dataset references from tool call arguments
fn extract_datasets(messages: &[ChatMessage]) -> Vec<ExportedDataset> {
    let mut datasets = Vec::new();
    let mut seen_names = std::collections::HashSet::new();

    for msg in messages {
        for tc in &msg.tool_calls {
            // Look for "dataset" field in tool call arguments
            if let Some(dataset_name) = tc.arguments.get("dataset").and_then(|v| v.as_str()) {
                if !seen_names.contains(dataset_name) {
                    seen_names.insert(dataset_name.to_string());
                    datasets.push(ExportedDataset {
                        name: dataset_name.to_string(),
                        rows: None,
                        cols: None,
                    });
                }
            }
        }
    }

    datasets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ChatMessage;

    #[test]
    fn test_export_empty_conversation() {
        let options = JsonExportOptions::default();
        let result = export_to_json(None, &[], &options);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("\"version\":\"1.0\""));
        assert!(json.contains("\"messages\":[]"));
    }

    #[test]
    fn test_export_with_messages() {
        let options = JsonExportOptions {
            pretty: true,
            include_tool_results: true,
            include_images: false,
        };

        let messages = vec![ChatMessage::user("Hello, can you help me?"), {
            let mut msg = ChatMessage::assistant_streaming();
            msg.content = "Of course! What do you need?".to_string();
            msg.is_streaming = false;
            msg
        }];

        let result = export_to_json(None, &messages, &options);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("Hello, can you help me?"));
        assert!(json.contains("Of course! What do you need?"));
        // `pretty: true` inserts a space after the colon — match either form.
        assert!(
            json.contains("\"role\":\"user\"") || json.contains("\"role\": \"user\""),
            "missing user role in output:\n{json}"
        );
        assert!(
            json.contains("\"role\":\"assistant\"") || json.contains("\"role\": \"assistant\""),
            "missing assistant role in output:\n{json}"
        );
    }
}
