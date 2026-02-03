//! HTML export functionality

use crate::api::Conversation;
use crate::state::ChatMessage;

use super::types::HtmlExportOptions;

/// Export a conversation and its messages to self-contained HTML
pub fn export_to_html(
    conversation: Option<&Conversation>,
    messages: &[ChatMessage],
    options: &HtmlExportOptions,
) -> String {
    let title = conversation
        .map(|c| c.title.as_str())
        .unwrap_or("Untitled Conversation");

    let created_at = conversation
        .map(|c| format_timestamp(&c.created_at))
        .unwrap_or_else(|| "Unknown".to_string());

    let css = if options.dark_theme {
        CSS_DARK
    } else {
        CSS_LIGHT
    };

    let messages_html = messages
        .iter()
        .filter(|m| !m.is_streaming)
        .map(|m| format_message_html(m, options))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="meta">Created: {created_at}</p>
        </header>
        <main class="messages">
{messages_html}
        </main>
        <footer>
            <p>Exported from <a href="https://github.com/prompt2analytics/prompt2analytics">prompt2analytics</a></p>
        </footer>
    </div>
</body>
</html>"#,
        title = escape_html(title),
        css = css,
        created_at = escape_html(&created_at),
        messages_html = messages_html
    )
}

/// Format a single message to HTML
fn format_message_html(msg: &ChatMessage, options: &HtmlExportOptions) -> String {
    let role_class = match msg.role.as_str() {
        "user" => "message user",
        "assistant" => "message assistant",
        _ => "message",
    };

    let role_display = match msg.role.as_str() {
        "user" => "User",
        "assistant" => "Assistant",
        _ => &msg.role,
    };

    let content_html = escape_html(&msg.content)
        .replace("\n\n", "</p><p>")
        .replace('\n', "<br>");

    let mut tool_calls_html = String::new();
    if options.include_tool_calls && !msg.tool_calls.is_empty() {
        for tc in &msg.tool_calls {
            let status_class = match tc.success {
                Some(true) => "success",
                Some(false) => "error",
                None => "pending",
            };

            let status_text = match tc.success {
                Some(true) => "completed",
                Some(false) => "failed",
                None => "pending",
            };

            let args_json = serde_json::to_string_pretty(&tc.arguments)
                .unwrap_or_else(|_| tc.arguments.to_string());

            let result_html = if let Some(ref result) = tc.result {
                let truncated = if result.len() > 2000 {
                    format!("{}... (truncated)", &result[..2000])
                } else {
                    result.clone()
                };
                format!(
                    r#"<div class="tool-result"><strong>Result:</strong><pre>{}</pre></div>"#,
                    escape_html(&truncated)
                )
            } else {
                String::new()
            };

            tool_calls_html.push_str(&format!(
                r#"<details class="tool-call {status_class}">
    <summary><code>{name}</code> <span class="status">({status_text})</span></summary>
    <div class="tool-args"><strong>Arguments:</strong><pre>{args}</pre></div>
    {result}
</details>"#,
                status_class = status_class,
                name = escape_html(&tc.name),
                status_text = status_text,
                args = escape_html(&args_json),
                result = result_html
            ));
        }
    }

    let mut images_html = String::new();
    if options.include_images && !msg.images.is_empty() {
        for (i, img) in msg.images.iter().enumerate() {
            let mime = if img.starts_with("/9j/") {
                "image/jpeg"
            } else {
                "image/png"
            };
            images_html.push_str(&format!(
                r#"<div class="image-container"><img src="data:{};base64,{}" alt="Image {}" /></div>"#,
                mime,
                img,
                i + 1
            ));
        }
    }

    format!(
        r#"            <div class="{role_class}">
                <div class="role">{role_display}</div>
                <div class="content"><p>{content}</p></div>
                {tool_calls}
                {images}
            </div>"#,
        role_class = role_class,
        role_display = role_display,
        content = content_html,
        tool_calls = tool_calls_html,
        images = images_html
    )
}

/// Escape HTML special characters
fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Format a timestamp string for display
fn format_timestamp(ts: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|_| ts.to_string())
}

/// Light theme CSS
const CSS_LIGHT: &str = r#"
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-user: #e3f2fd;
            --bg-assistant: #f5f5f5;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --border-color: #dee2e6;
            --accent-color: #ea580c;
            --success-color: #198754;
            --error-color: #dc3545;
            --code-bg: #f1f3f5;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        header h1 {
            color: var(--text-primary);
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }

        .meta {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .messages {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .message {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .message.user {
            background: var(--bg-user);
            border-left: 4px solid #2196f3;
        }

        .message.assistant {
            background: var(--bg-assistant);
            border-left: 4px solid var(--accent-color);
        }

        .role {
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        .content p {
            margin-bottom: 0.75rem;
        }

        .content p:last-child {
            margin-bottom: 0;
        }

        .tool-call {
            margin-top: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            overflow: hidden;
        }

        .tool-call summary {
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tool-call summary code {
            background: var(--code-bg);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.875rem;
        }

        .tool-call .status {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .tool-call.success .status {
            color: var(--success-color);
        }

        .tool-call.error .status {
            color: var(--error-color);
        }

        .tool-args, .tool-result {
            padding: 1rem;
        }

        .tool-args pre, .tool-result pre {
            background: var(--code-bg);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.8125rem;
            margin-top: 0.5rem;
        }

        .image-container {
            margin-top: 1rem;
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        footer a {
            color: var(--accent-color);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }
"#;

/// Dark theme CSS
const CSS_DARK: &str = r#"
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-user: #1e3a5f;
            --bg-assistant: #2d2d2d;
            --text-primary: #e4e4e7;
            --text-secondary: #a1a1aa;
            --border-color: #3f3f46;
            --accent-color: #fb923c;
            --success-color: #4ade80;
            --error-color: #f87171;
            --code-bg: #27272a;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }

        header {
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        header h1 {
            color: var(--text-primary);
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }

        .meta {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .messages {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .message {
            padding: 1rem 1.25rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .message.user {
            background: var(--bg-user);
            border-left: 4px solid #3b82f6;
        }

        .message.assistant {
            background: var(--bg-assistant);
            border-left: 4px solid var(--accent-color);
        }

        .role {
            font-weight: 600;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        .content p {
            margin-bottom: 0.75rem;
        }

        .content p:last-child {
            margin-bottom: 0;
        }

        .tool-call {
            margin-top: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            overflow: hidden;
        }

        .tool-call summary {
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tool-call summary code {
            background: var(--code-bg);
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.875rem;
        }

        .tool-call .status {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .tool-call.success .status {
            color: var(--success-color);
        }

        .tool-call.error .status {
            color: var(--error-color);
        }

        .tool-args, .tool-result {
            padding: 1rem;
        }

        .tool-args pre, .tool-result pre {
            background: var(--code-bg);
            padding: 0.75rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.8125rem;
            margin-top: 0.5rem;
        }

        .image-container {
            margin-top: 1rem;
        }

        .image-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        footer a {
            color: var(--accent-color);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ChatMessage;

    #[test]
    fn test_export_empty_conversation() {
        let options = HtmlExportOptions::default();
        let result = export_to_html(None, &[], &options);

        assert!(result.contains("<!DOCTYPE html>"));
        assert!(result.contains("Untitled Conversation"));
        assert!(result.contains("prompt2analytics"));
    }

    #[test]
    fn test_export_with_messages() {
        let options = HtmlExportOptions::default();

        let messages = vec![ChatMessage::user("Hello"), {
            let mut msg = ChatMessage::assistant_streaming();
            msg.content = "Hi there!".to_string();
            msg.is_streaming = false;
            msg
        }];

        let result = export_to_html(None, &messages, &options);

        assert!(result.contains("class=\"message user\""));
        assert!(result.contains("class=\"message assistant\""));
        assert!(result.contains("Hello"));
        assert!(result.contains("Hi there!"));
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(
            escape_html("<script>alert('xss')</script>"),
            "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;"
        );
    }

    #[test]
    fn test_dark_theme() {
        let options = HtmlExportOptions {
            dark_theme: true,
            ..Default::default()
        };

        let result = export_to_html(None, &[], &options);
        assert!(result.contains("--bg-primary: #1a1a1a"));
    }
}
