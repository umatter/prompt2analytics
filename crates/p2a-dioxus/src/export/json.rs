//! JSON export functionality

use chrono::Utc;

use crate::api::Conversation;
use crate::state::ChatMessage;

use super::types::{
    ExportedConversation, ExportedConversationMeta, ExportedDataset, ExportedMessage,
    ExportedToolCall, JsonExportOptions,
};

/// Shortest id that is redacted from exported text. Session and
/// conversation ids are UUIDs (36 bytes); the floor keeps a degenerate id
/// from blanking out ordinary words.
const MIN_REDACTED_SESSION_ID_LEN: usize = 16;

/// Export a conversation and its messages to JSON format.
///
/// The export is meant to be shared, so it carries no live identifiers:
/// neither the session id (the only credential for a visitor's datasets on
/// the open demo) nor the backend conversation id (which alone grants
/// read, update and delete on that conversation) is written, and any
/// occurrence of either in message text, tool arguments or results is
/// replaced by `[redacted]`. The conversation and message ids are
/// export-local, not the server's.
pub fn export_to_json(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &JsonExportOptions,
) -> Result<String, String> {
    let exported = build_exported_conversation(conversation, messages, options);

    let json = if options.pretty {
        serde_json::to_string_pretty(&exported)
    } else {
        serde_json::to_string(&exported)
    }
    .map_err(|e| format!("JSON serialization error: {}", e))?;

    let ids = conversation
        .map(|c| [c.session_id.as_str(), c.id.as_str()])
        .unwrap_or_default();
    Ok(redact_ids(json, &ids))
}

/// Replace every occurrence of each id in the serialized export with
/// `[redacted]`. Ids shorter than [`MIN_REDACTED_SESSION_ID_LEN`] are left
/// alone.
///
/// Each id is matched in its JSON-escaped form, which is how it appears
/// inside the serialized strings; for a UUID that is the id itself. Longer
/// ids go first, so an id that contains another is not left half-redacted.
fn redact_ids(mut json: String, ids: &[&str]) -> String {
    let mut ids: Vec<&str> = ids
        .iter()
        .copied()
        .filter(|id| id.len() >= MIN_REDACTED_SESSION_ID_LEN)
        .collect();
    ids.sort_by_key(|id| std::cmp::Reverse(id.len()));
    for id in ids {
        let quoted = serde_json::to_string(id).unwrap_or_default();
        let escaped = quoted
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(id);
        json = json.replace(escaped, "[redacted]");
    }
    json
}

/// Build the exported conversation structure
fn build_exported_conversation(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &JsonExportOptions,
) -> ExportedConversation {
    let now = Utc::now();

    // Build conversation metadata. The id is always export-local: the
    // server's conversation id grants access to the conversation, so it is
    // not written to a file meant for sharing.
    let export_id = uuid::Uuid::new_v4().to_string();
    let conversation_meta = if let Some(conv) = conversation {
        ExportedConversationMeta {
            id: export_id,
            title: conv.title.clone(),
            created_at: conv.created_at.clone(),
            updated_at: conv.updated_at.clone(),
        }
    } else {
        ExportedConversationMeta {
            id: export_id,
            title: "Untitled Conversation".to_string(),
            created_at: now.to_rfc3339(),
            updated_at: now.to_rfc3339(),
        }
    };

    // Convert messages, numbering them after dropping incomplete streaming
    // messages.
    let exported_messages: Vec<ExportedMessage> = messages
        .iter()
        .filter(|m| !m.is_streaming)
        .enumerate()
        .map(|(index, m)| convert_message(m, index, options))
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

/// Convert a ChatMessage to ExportedMessage.
///
/// The message id is `msg-<index + 1>`: server message ids are capabilities
/// on the tool-call history endpoint. Tool-call ids come from the LLM
/// provider and grant nothing, so they are kept.
fn convert_message(
    msg: &ChatMessage,
    index: usize,
    options: &JsonExportOptions,
) -> ExportedMessage {
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
        id: format!("msg-{}", index + 1),
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
    use crate::api::Conversation;
    use crate::state::{ChatMessage, ToolCallInfo};

    #[test]
    fn json_export_omits_session_and_backend_ids() {
        let sid = "b01dc9ce-1111-2222-3333-444455556666";
        let conv_id = "c0a7e1d2-aaaa-bbbb-cccc-ddddeeeeffff";
        let conv = Conversation {
            id: conv_id.into(),
            session_id: sid.into(),
            title: "Wage study".into(),
            created_at: "2026-10-01T00:00:00Z".into(),
            updated_at: "2026-10-01T00:00:00Z".into(),
            is_archived: false,
            message_count: 2,
            last_message_preview: None,
        };
        let mut user = ChatMessage::user("hi");
        user.id = "msg-backend-42".into();
        let mut asst = ChatMessage::assistant_streaming();
        asst.is_streaming = false;
        asst.id = "msg-backend-43".into();
        asst.content = format!("session {sid} is active");
        asst.tool_calls.push(ToolCallInfo {
            id: "call_1".into(),
            name: "head_dataset".into(),
            arguments: serde_json::json!({"dataset": "wages"}),
            result: Some(format!("{{\"conversation_id\": \"{conv_id}\"}}")),
            success: Some(true),
            is_expanded: false,
        });
        let opts = JsonExportOptions {
            pretty: false,
            include_tool_results: true,
            include_images: true,
        };
        let json = export_to_json(Some(&conv), &[user, asst], &opts).unwrap();
        for leaked in [
            sid,
            conv_id,
            "msg-backend-42",
            "msg-backend-43",
            "session_id",
        ] {
            assert!(!json.contains(leaked), "{leaked} leaked: {json}");
        }
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["conversation"]["title"], "Wage study");
        assert_eq!(v["messages"][0]["id"], "msg-1");
        assert_eq!(v["messages"][1]["id"], "msg-2");
        assert_eq!(v["messages"][1]["content"], "session [redacted] is active");
        assert_eq!(v["messages"][1]["tool_calls"][0]["name"], "head_dataset");
        assert_eq!(
            v["messages"][1]["tool_calls"][0]["result"],
            "{\"conversation_id\": \"[redacted]\"}"
        );
    }

    #[test]
    fn redact_ids_handles_short_nested_and_escaped_ids() {
        let long = "0123456789abcdef-long";
        let inner = "0123456789abcdef";
        let json = serde_json::to_string(&serde_json::json!({
            "a": format!("{long} / {inner} / short"),
            "b": "id\"with-a-quote-xyz",
        }))
        .unwrap();
        let out = redact_ids(json, &[inner, "short", long, "id\"with-a-quote-xyz"]);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["a"], "[redacted] / [redacted] / short");
        assert_eq!(v["b"], "[redacted]");
    }

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
