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
}

/// ToolCall display component - expandable card showing tool invocation
#[component]
pub fn ToolCallDisplay(props: ToolCallProps) -> Element {
    let mut is_expanded = use_signal(|| false);
    let mut result_expanded = use_signal(|| false);
    let mut copied = use_signal(|| false);

    let (status_bg, status_text, status_icon) = match props.success {
        Some(true) => (
            "bg-green-100 dark:bg-green-900/30",
            "text-green-700 dark:text-green-400",
            rsx! {
                svg {
                    class: "w-4 h-4",
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "2",
                        d: "M5 13l4 4L19 7"
                    }
                }
            },
        ),
        Some(false) => (
            "bg-red-100 dark:bg-red-900/30",
            "text-red-700 dark:text-red-400",
            rsx! {
                svg {
                    class: "w-4 h-4",
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "2",
                        d: "M6 18L18 6M6 6l12 12"
                    }
                }
            },
        ),
        None => (
            "bg-orange-100 dark:bg-orange-900/30",
            "text-orange-700 dark:text-orange-400",
            rsx! {
                svg {
                    class: "w-4 h-4 animate-spin",
                    fill: "none",
                    view_box: "0 0 24 24",
                    circle {
                        class: "opacity-25",
                        cx: "12",
                        cy: "12",
                        r: "10",
                        stroke: "currentColor",
                        stroke_width: "4"
                    }
                    path {
                        class: "opacity-75",
                        fill: "currentColor",
                        d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    }
                }
            },
        ),
    };

    let args_pretty = serde_json::to_string_pretty(&props.arguments).unwrap_or_default();
    let expanded = *is_expanded.read();

    rsx! {
        div { class: "my-3 rounded-lg border border-gray-300 dark:border-gray-700 overflow-hidden shadow-sm hover:shadow-md transition-shadow bg-white dark:bg-gray-800",
            // Header (clickable)
            button {
                class: "w-full px-4 py-3 flex items-center justify-between {status_bg} cursor-pointer",
                onclick: move |_| {
                    let current = *is_expanded.read();
                    is_expanded.set(!current);
                },

                div { class: "flex items-center gap-3",
                    // Status icon
                    span { class: "{status_text}",
                        {status_icon}
                    }
                    // Tool name
                    span { class: "font-medium text-gray-900 dark:text-white",
                        "{props.name}"
                    }
                    // Small badge showing it's a tool
                    span { class: "px-2 py-0.5 text-xs font-medium rounded bg-orange-100 dark:bg-orange-800 text-orange-700 dark:text-orange-300",
                        "tool"
                    }
                }

                // Expand/collapse icon
                svg {
                    class: "w-5 h-5 text-gray-500 transition-transform duration-200",
                    class: if expanded { "rotate-180" } else { "" },
                    fill: "none",
                    stroke: "currentColor",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        stroke_width: "2",
                        d: "M19 9l-7 7-7-7"
                    }
                }
            }

            // Content (expandable)
            if expanded {
                div { class: "px-4 py-3 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700",
                    // Arguments section
                    div { class: "mb-4",
                        div { class: "flex items-center gap-2 mb-2",
                            svg {
                                class: "w-4 h-4 text-gray-400",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                                }
                            }
                            span { class: "text-sm font-medium text-gray-600 dark:text-gray-400", "Arguments" }
                        }
                        pre { class: "text-sm bg-gray-50 dark:bg-gray-800 p-3 rounded-lg overflow-x-auto font-mono text-gray-800 dark:text-gray-200",
                            "{args_pretty}"
                        }
                    }

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
                                        let escaped = result_for_copy.replace('\\', "\\\\").replace('\'', "\\'").replace('\n', "\\n");
                                        let js = format!("navigator.clipboard.writeText('{}')", escaped);
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

                            rsx! {
                                div {
                                    // Header with copy button
                                    div { class: "flex items-center justify-between mb-2",
                                        div { class: "flex items-center gap-2",
                                            svg {
                                                class: "w-4 h-4 text-gray-400",
                                                fill: "none",
                                                stroke: "currentColor",
                                                view_box: "0 0 24 24",
                                                path {
                                                    stroke_linecap: "round",
                                                    stroke_linejoin: "round",
                                                    stroke_width: "2",
                                                    d: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                                                }
                                            }
                                            span { class: "text-sm font-medium text-gray-600 dark:text-gray-400", "Result" }
                                            // Format badge
                                            span {
                                                class: match format {
                                                    ResultFormat::Json => "px-1.5 py-0.5 text-xs rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
                                                    ResultFormat::Table => "px-1.5 py-0.5 text-xs rounded bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
                                                    ResultFormat::PlainText => "px-1.5 py-0.5 text-xs rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400",
                                                },
                                                match format {
                                                    ResultFormat::Json => "JSON",
                                                    ResultFormat::Table => "Table",
                                                    ResultFormat::PlainText => "Text",
                                                }
                                            }
                                        }
                                        // Copy button
                                        button {
                                            class: "p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                                            onclick: handle_copy,
                                            title: "Copy result",
                                            if *copied.read() {
                                                svg {
                                                    class: "w-4 h-4 text-green-500",
                                                    fill: "none",
                                                    stroke: "currentColor",
                                                    view_box: "0 0 24 24",
                                                    path {
                                                        stroke_linecap: "round",
                                                        stroke_linejoin: "round",
                                                        stroke_width: "2",
                                                        d: "M5 13l4 4L19 7"
                                                    }
                                                }
                                            } else {
                                                svg {
                                                    class: "w-4 h-4",
                                                    fill: "none",
                                                    stroke: "currentColor",
                                                    view_box: "0 0 24 24",
                                                    path {
                                                        stroke_linecap: "round",
                                                        stroke_linejoin: "round",
                                                        stroke_width: "2",
                                                        d: "M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Result content with format-specific styling
                                    div { class: "relative",
                                        match format {
                                            ResultFormat::Json => rsx! {
                                                pre { class: "text-sm bg-slate-800 dark:bg-slate-900 p-3 rounded-lg overflow-x-auto font-mono text-slate-200 whitespace-pre-wrap max-h-96 overflow-y-auto",
                                                    // Try to pretty-print JSON
                                                    {
                                                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&display_text) {
                                                            serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| display_text.clone())
                                                        } else {
                                                            display_text.clone()
                                                        }
                                                    }
                                                }
                                            },
                                            ResultFormat::Table => rsx! {
                                                div { class: "overflow-x-auto",
                                                    pre { class: "text-sm bg-gray-50 dark:bg-gray-800 p-3 rounded-lg font-mono text-gray-800 dark:text-gray-200 whitespace-pre max-h-96 overflow-y-auto",
                                                        "{display_text}"
                                                    }
                                                }
                                            },
                                            ResultFormat::PlainText => rsx! {
                                                pre { class: "text-sm bg-gray-50 dark:bg-gray-800 p-3 rounded-lg overflow-x-auto font-mono text-gray-800 dark:text-gray-200 whitespace-pre-wrap max-h-96 overflow-y-auto",
                                                    "{display_text}"
                                                }
                                            },
                                        }

                                        // Show more/less button
                                        if has_more {
                                            div { class: "mt-2 flex justify-center",
                                                button {
                                                    class: "px-3 py-1 text-xs font-medium text-teal-600 dark:text-teal-400 bg-teal-50 dark:bg-teal-900/30 rounded-full hover:bg-teal-100 dark:hover:bg-teal-900/50 transition-colors flex items-center gap-1",
                                                    onclick: move |_| {
                                                        let current = *result_expanded.read();
                                                        result_expanded.set(!current);
                                                    },
                                                    if is_result_expanded {
                                                        svg {
                                                            class: "w-3 h-3",
                                                            fill: "none",
                                                            stroke: "currentColor",
                                                            view_box: "0 0 24 24",
                                                            path {
                                                                stroke_linecap: "round",
                                                                stroke_linejoin: "round",
                                                                stroke_width: "2",
                                                                d: "M5 15l7-7 7 7"
                                                            }
                                                        }
                                                        "Show less"
                                                    } else {
                                                        svg {
                                                            class: "w-3 h-3",
                                                            fill: "none",
                                                            stroke: "currentColor",
                                                            view_box: "0 0 24 24",
                                                            path {
                                                                stroke_linecap: "round",
                                                                stroke_linejoin: "round",
                                                                stroke_width: "2",
                                                                d: "M19 9l-7 7-7-7"
                                                            }
                                                        }
                                                        "Show more"
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
        }
    }
}
