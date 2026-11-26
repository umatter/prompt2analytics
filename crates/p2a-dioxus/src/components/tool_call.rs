//! ToolCall display component

use dioxus::prelude::*;

/// Result format detected from content
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResultFormat {
    /// Plain text (default)
    PlainText,
    /// JSON object or array
    Json,
    /// Markdown-style table (lines with | separators)
    Table,
}

/// Detect the format of a result string
fn detect_result_format(result: &str) -> ResultFormat {
    let trimmed = result.trim();

    // Check for JSON
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        // Verify it's valid JSON
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return ResultFormat::Json;
        }
    }

    // Check for table (multiple lines with | separators)
    let lines: Vec<&str> = trimmed.lines().collect();
    if lines.len() >= 2 {
        let has_pipes = lines.iter().take(3).all(|line| line.contains('|'));
        let has_separator = lines.iter().any(|line| {
            line.chars()
                .all(|c| c == '-' || c == '|' || c == ' ' || c == ':')
                && line.contains('-')
        });
        if has_pipes && has_separator {
            return ResultFormat::Table;
        }
    }

    ResultFormat::PlainText
}

/// Get a preview of the result (first few lines)
fn get_result_preview(result: &str, max_lines: usize) -> (String, bool) {
    let lines: Vec<&str> = result.lines().collect();
    let has_more = lines.len() > max_lines;
    let preview = lines
        .into_iter()
        .take(max_lines)
        .collect::<Vec<_>>()
        .join("\n");
    (preview, has_more)
}

/// A tool call's status as words: its outcome, or, without one, "status
/// unknown" once the call has ended (an older server sends no status) and
/// "running" while it runs.
pub fn status_label(success: Option<bool>, finished: bool) -> &'static str {
    match (success, finished) {
        (Some(true), _) => "completed",
        (Some(false), _) => "failed",
        (None, true) => "status unknown",
        (None, false) => "running",
    }
}

/// The arguments of a call on one line, as the CLI would print them:
/// `key=value` pairs in the map's order, strings bare unless they hold a
/// space (then quoted), arrays joined by commas, nested objects as `{…}`;
/// cut to `max_chars` characters with `...`.
pub fn args_summary(arguments: &serde_json::Value, max_chars: usize) -> String {
    fn scalar(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::String(s) if s.contains(char::is_whitespace) || s.is_empty() => {
                format!("\"{s}\"")
            }
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(items) => {
                items.iter().map(scalar).collect::<Vec<_>>().join(",")
            }
            serde_json::Value::Object(_) => "{…}".to_string(),
            other => other.to_string(),
        }
    }
    let line = match arguments {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}={}", scalar(v)))
            .collect::<Vec<_>>()
            .join(" "),
        serde_json::Value::Null => String::new(),
        other => scalar(other),
    };
    p2a_text::ellipsize(&line, max_chars).into_owned()
}

/// The first line of a failed call's result, shown under its tree line
/// while the call is collapsed.
fn first_line(result: &str) -> &str {
    result.trim().lines().next().unwrap_or("")
}

/// Props for ToolCallDisplay
#[derive(Props, Clone, PartialEq)]
pub struct ToolCallProps {
    /// Tool call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool arguments as JSON
    pub arguments: serde_json::Value,
    /// Tool result (if completed)
    #[props(default)]
    pub result: Option<String>,
    /// Whether the tool call succeeded
    #[props(default)]
    pub success: Option<bool>,
    /// Whether the call has ended (with no outcome: "status unknown")
    #[props(default)]
    pub finished: bool,
    /// Whether this is the last call of its message (drawn `└─`, not `├─`)
    #[props(default)]
    pub last: bool,
}

/// ToolCall display component - one line of the tool-call process tree,
/// opening to the call's arguments and result
#[component]
pub fn ToolCallDisplay(props: ToolCallProps) -> Element {
    let mut is_expanded = use_signal(|| false);
    let mut result_expanded = use_signal(|| false);
    let mut copied = use_signal(|| false);

    let status_word = status_label(props.success, props.finished);
    let args_pretty = serde_json::to_string_pretty(&props.arguments).unwrap_or_default();
    let args_line = args_summary(&props.arguments, 80);
    let expanded = *is_expanded.read();
    let branch = if props.last { "└─ " } else { "├─ " };
    let stem_class = if props.last {
        "k-call-body k-call-last"
    } else {
        "k-call-body"
    };
    let failure_line = match (props.success, props.result.as_deref()) {
        (Some(false), Some(r)) if !expanded => Some(first_line(r).to_string()),
        _ => None,
    };

    rsx! {
        div { class: "k-call",
            // The tree line (clickable)
            button {
                class: "k-call-line",
                aria_expanded: if expanded { "true" } else { "false" },
                title: "{props.name}: {status_word}",
                onclick: move |_| {
                    let current = *is_expanded.read();
                    is_expanded.set(!current);
                },
                span { class: "g", "{branch}" }
                span { class: "tn", "{props.name}" }
                if !args_line.is_empty() {
                    span { class: "k-call-args", "{args_line}" }
                }
                span { class: "k-lead", aria_hidden: "true" }
                match (props.success, props.finished) {
                    (Some(true), _) => rsx! { span { class: "k-call-status", "done" } },
                    (Some(false), _) => rsx! { span { class: "k-call-status bad", "failed" } },
                    (None, true) => rsx! { span { class: "k-call-status g", "status unknown" } },
                    (None, false) => rsx! { span { class: "k-call-status run k-spin", " running" } },
                }
            }

            // A failed call's error, while collapsed
            if let Some(ref why) = failure_line {
                div { class: "{stem_class}",
                    span { class: "g", "└ " }
                    span { class: "k-why", "{why}" }
                }
            }

            // Content (expandable)
            if expanded {
                div { class: "{stem_class}",
                    // Arguments section
                    div { class: "k-call-sec", "Arguments" }
                    pre { class: "k-call-pre", "{args_pretty}" }

                    // Result section (if available)
                    if let Some(ref result) = props.result {
                        {
                            let format = detect_result_format(result);
                            let (preview, has_more) = get_result_preview(result, 8);
                            let is_result_expanded = *result_expanded.read();
                            let display_text = if is_result_expanded || !has_more {
                                result.clone()
                            } else {
                                preview.clone()
                            };
                            let result_for_copy = result.clone();

                            // Handle copy
                            let handle_copy = {
                                move |_| {
                                    #[cfg(target_arch = "wasm32")]
                                    {
                                        if let Some(window) = web_sys::window() {
                                            let navigator = window.navigator();
                                            let clipboard = navigator.clipboard();
                                            let text = result_for_copy.clone();
                                            wasm_bindgen_futures::spawn_local(async move {
                                                let _ = wasm_bindgen_futures::JsFuture::from(
                                                    clipboard.write_text(&text)
                                                ).await;
                                            });
                                        }
                                    }
                                    #[cfg(not(target_arch = "wasm32"))]
                                    {
                                        let js = format!(
                                            "navigator.clipboard.writeText('{}')",
                                            crate::utils::js_single_quoted(&result_for_copy)
                                        );
                                        dioxus::document::eval(&js);
                                    }
                                    copied.set(true);
                                    // Reset after 2 seconds
                                    spawn(async move {
                                        #[cfg(all(not(target_arch = "wasm32"), any(feature = "desktop", feature = "mobile")))]
                                        {
                                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                        }
                                        #[cfg(target_arch = "wasm32")]
                                        {
                                            gloo_timers::future::TimeoutFuture::new(2_000).await;
                                        }
                                        copied.set(false);
                                    });
                                }
                            };

                            let shown = match format {
                                ResultFormat::Json => {
                                    // Try to pretty-print JSON
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&display_text) {
                                        serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| display_text.clone())
                                    } else {
                                        display_text.clone()
                                    }
                                }
                                _ => display_text.clone(),
                            };
                            let pre_class = match format {
                                // A table keeps its columns: scroll, do not wrap.
                                ResultFormat::Table => "k-call-pre k-call-result k-call-nowrap",
                                _ => "k-call-pre k-call-result",
                            };

                            rsx! {
                                div {
                                    // Header: label, format and copy
                                    div { class: "k-call-sec",
                                        span { "Result" }
                                        span { class: "g",
                                            match format {
                                                ResultFormat::Json => "json",
                                                ResultFormat::Table => "table",
                                                ResultFormat::PlainText => "text",
                                            }
                                        }
                                        button {
                                            class: "k-call-act",
                                            onclick: handle_copy,
                                            title: "Copy result",
                                            if *copied.read() { "copied" } else { "copy" }
                                        }
                                    }

                                    pre { class: "{pre_class}", "{shown}" }

                                    // Show more/less button
                                    if has_more {
                                        button {
                                            class: "k-call-act",
                                            onclick: move |_| {
                                                let current = *result_expanded.read();
                                                result_expanded.set(!current);
                                            },
                                            if is_result_expanded { "Show less" } else { "Show more" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_label_table() {
        assert_eq!(status_label(Some(true), true), "completed");
        assert_eq!(status_label(Some(false), true), "failed");
        assert_eq!(status_label(None, true), "status unknown");
        assert_eq!(status_label(None, false), "running");
        assert_eq!(status_label(Some(true), false), "completed");
        assert_eq!(status_label(Some(false), false), "failed");
    }

    #[test]
    fn args_summary_prints_one_cli_line() {
        let args = serde_json::json!({
            "dataset": "grunfeld",
            "y": "inv",
            "x": ["value", "capital"],
            "path": "my data.csv",
            "robust": true,
            "level": 0.95,
            "opts": {"a": 1}
        });
        let line = args_summary(&args, 200);
        assert!(line.contains("dataset=grunfeld"), "{line}");
        assert!(line.contains("x=value,capital"), "{line}");
        assert!(line.contains("path=\"my data.csv\""), "{line}");
        assert!(line.contains("robust=true"), "{line}");
        assert!(line.contains("level=0.95"), "{line}");
        assert!(line.contains("opts={…}"), "{line}");
        assert_eq!(args_summary(&serde_json::Value::Null, 80), "");
        let cut = args_summary(&args, 12);
        assert!(cut.chars().count() <= 12, "{cut}");
    }
}
