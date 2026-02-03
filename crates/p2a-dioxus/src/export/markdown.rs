//! Markdown export functionality

use crate::api::Conversation;
use crate::state::ChatMessage;

use super::types::MarkdownExportOptions;

/// Export a conversation and its messages to Markdown format
pub fn export_to_markdown(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &MarkdownExportOptions,
) -> String {
    let mut output = String::new();

    // Title
    let title = conversation
        .map(|c| c.title.as_str())
        .unwrap_or("Untitled Conversation");
    output.push_str(&format!("# {}\n\n", escape_markdown(title)));

    // Metadata
    if let Some(conv) = conversation {
        output.push_str(&format!(
            "*Created: {}*\n\n",
            format_timestamp(&conv.created_at)
        ));
    }

    output.push_str("---\n\n");

    // Messages
    for msg in messages.iter().filter(|m| !m.is_streaming) {
        output.push_str(&format_message(msg, options));
        output.push_str("\n---\n\n");
    }

    // Footer
    output.push_str("\n*Exported from prompt2analytics*\n");

    output
}

/// Format a single message to Markdown
fn format_message(msg: &ChatMessage, options: &MarkdownExportOptions) -> String {
    let mut output = String::new();

    // Role header with optional timestamp
    let role_display = match msg.role.as_str() {
        "user" => "**User**",
        "assistant" => "**Assistant**",
        _ => &msg.role,
    };

    if options.include_timestamps {
        output.push_str(&format!(
            "{} *{}*\n\n",
            role_display,
            msg.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        ));
    } else {
        output.push_str(&format!("{}\n\n", role_display));
    }

    // Message content
    output.push_str(&msg.content);
    output.push_str("\n\n");

    // Tool calls in collapsible sections
    if options.include_tool_calls && !msg.tool_calls.is_empty() {
        for tc in &msg.tool_calls {
            let status_indicator = match tc.success {
                Some(true) => "completed",
                Some(false) => "failed",
                None => "pending",
            };

            output.push_str(&format!(
                "<details>\n<summary>Tool: <code>{}</code> ({})</summary>\n\n",
                escape_markdown(&tc.name),
                status_indicator
            ));

            // Arguments
            output.push_str("**Arguments:**\n```json\n");
            if let Ok(pretty) = serde_json::to_string_pretty(&tc.arguments) {
                output.push_str(&pretty);
            } else {
                output.push_str(&tc.arguments.to_string());
            }
            output.push_str("\n```\n\n");

            // Result if present
            if let Some(ref result) = tc.result {
                output.push_str("**Result:**\n```\n");
                // Truncate very long results
                if result.len() > 2000 {
                    output.push_str(&result[..2000]);
                    output.push_str("\n... (truncated)");
                } else {
                    output.push_str(result);
                }
                output.push_str("\n```\n");
            }

            output.push_str("\n</details>\n\n");
        }
    }

    // Images
    if options.include_images && !msg.images.is_empty() {
        for (i, img) in msg.images.iter().enumerate() {
            // Determine MIME type (assume PNG if not specified)
            let mime = if img.starts_with("/9j/") {
                "image/jpeg"
            } else {
                "image/png"
            };
            output.push_str(&format!(
                "![Image {}](data:{};base64,{})\n\n",
                i + 1,
                mime,
                img
            ));
        }
    }

    output
}

/// Escape special Markdown characters
fn escape_markdown(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('*', "\\*")
        .replace('_', "\\_")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace('`', "\\`")
        .replace('#', "\\#")
}

/// Format a timestamp string for display
fn format_timestamp(ts: &str) -> String {
    // Try to parse and reformat, or just return as-is
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|_| ts.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ChatMessage;

    #[test]
    fn test_export_empty_conversation() {
        let options = MarkdownExportOptions::default();
        let result = export_to_markdown(None, &[], &options);

        assert!(result.contains("# Untitled Conversation"));
        assert!(result.contains("Exported from prompt2analytics"));
    }

    #[test]
    fn test_export_with_messages() {
        let options = MarkdownExportOptions {
            include_tool_calls: false,
            include_images: false,
            include_timestamps: false,
        };

        let messages = vec![ChatMessage::user("Hello"), {
            let mut msg = ChatMessage::assistant_streaming();
            msg.content = "Hi there!".to_string();
            msg.is_streaming = false;
            msg
        }];

        let result = export_to_markdown(None, &messages, &options);

        assert!(result.contains("**User**"));
        assert!(result.contains("Hello"));
        assert!(result.contains("**Assistant**"));
        assert!(result.contains("Hi there!"));
    }

    #[test]
    fn test_escape_markdown() {
        assert_eq!(escape_markdown("Hello *world*"), "Hello \\*world\\*");
        assert_eq!(escape_markdown("[link]"), "\\[link\\]");
    }
}
