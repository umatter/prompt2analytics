//! ToolCall display component

use dioxus::prelude::*;

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
            "bg-blue-100 dark:bg-blue-900/30",
            "text-blue-700 dark:text-blue-400",
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
        div { class: "my-3 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm hover:shadow-md transition-shadow",
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
                    span { class: "px-2 py-0.5 text-xs font-medium rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300",
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
                div { class: "px-4 py-3 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700",
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
                        pre { class: "text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded-lg overflow-x-auto font-mono text-gray-800 dark:text-gray-200",
                            "{args_pretty}"
                        }
                    }

                    // Result section (if available)
                    if let Some(ref result) = props.result {
                        div {
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
                                        d: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                                    }
                                }
                                span { class: "text-sm font-medium text-gray-600 dark:text-gray-400", "Result" }
                            }
                            pre { class: "text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded-lg overflow-x-auto font-mono text-gray-800 dark:text-gray-200 whitespace-pre-wrap max-h-96 overflow-y-auto",
                                "{result}"
                            }
                        }
                    }
                }
            }
        }
    }
}
