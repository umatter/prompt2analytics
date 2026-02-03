//! Message component for displaying chat messages

use dioxus::prelude::*;

use crate::state::chat::ChatMessage;
use crate::utils::render_markdown;

/// Props for the Message component
#[derive(Props, Clone, PartialEq)]
pub struct MessageProps {
    /// The message to display
    pub message: ChatMessage,
}

impl PartialEq for ChatMessage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.content == other.content
            && self.is_streaming == other.is_streaming
            && self.images.len() == other.images.len()
            && self.tool_calls.len() == other.tool_calls.len()
    }
}

/// Render message content based on role
fn render_content(message: &ChatMessage) -> Element {
    if message.role == "user" {
        rsx! {
            p { class: "text-white whitespace-pre-wrap",
                "{message.content}"
            }
        }
    } else {
        let content = &message.content;
        if message.is_streaming {
            rsx! {
                div { class: "text-gray-800 dark:text-gray-200",
                    {render_markdown(content)}
                    span { class: "inline-block w-2 h-4 ml-0.5 bg-teal-500 animate-pulse" }
                }
            }
        } else {
            render_markdown(content)
        }
    }
}

/// Message component
#[component]
pub fn Message(props: MessageProps) -> Element {
    let message = &props.message;
    let is_user = message.role == "user";

    let timestamp = message.timestamp.format("%H:%M").to_string();

    if is_user {
        // User message - right aligned, teal background
        rsx! {
            div { class: "flex justify-end mb-4 message-enter",
                div { class: "max-w-[80%] rounded-2xl rounded-tr-sm px-4 py-3 bg-teal-600 text-white shadow-sm",
                    // Message content
                    {render_content(message)}

                    // Timestamp
                    div { class: "text-xs text-teal-200 mt-2 text-right",
                        "{timestamp}"
                    }
                }
            }
        }
    } else {
        // Assistant message - left aligned, gray background
        rsx! {
            div { class: "flex justify-start mb-4 message-enter",
                div { class: "max-w-[85%]",
                    // Tools Used indicator (orange for Rust/p2a-core branding)
                    if !message.tool_calls.is_empty() {
                        div { class: "mb-2 px-3 py-2 rounded-lg bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-700",
                            div { class: "flex items-center gap-2 flex-wrap",
                                // Rust gear icon
                                svg {
                                    class: "w-4 h-4 text-orange-600 dark:text-orange-400 flex-shrink-0",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                                    }
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                    }
                                }
                                span { class: "text-xs font-semibold text-orange-700 dark:text-orange-300",
                                    "Rust Analytics:"
                                }
                                // Tool chips
                                for tool_call in message.tool_calls.iter() {
                                    span {
                                        key: "{tool_call.id}",
                                        class: "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-200",
                                        "{tool_call.name}"
                                    }
                                }
                            }
                        }
                    }

                    // Main message bubble
                    div { class: "rounded-2xl rounded-tl-sm px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
                        // Message content
                        {render_content(message)}

                        // Display images if any
                        if !message.images.is_empty() {
                            div { class: "mt-3 space-y-2",
                                for (idx , image) in message.images.iter().enumerate() {
                                    img {
                                        key: "{idx}",
                                        src: "data:image/png;base64,{image}",
                                        class: "max-w-full rounded-lg shadow-md",
                                        alt: "Visualization"
                                    }
                                }
                            }
                        }

                        // Timestamp
                        div { class: "text-xs text-gray-500 dark:text-gray-400 mt-2",
                            "{timestamp}"
                        }
                    }

                    // Tool call details (expandable cards) displayed below
                    if !message.tool_calls.is_empty() {
                        div { class: "mt-2 space-y-2",
                            for tool_call in message.tool_calls.iter() {
                                crate::components::ToolCallDisplay {
                                    key: "{tool_call.id}",
                                    id: tool_call.id.clone(),
                                    name: tool_call.name.clone(),
                                    arguments: tool_call.arguments.clone(),
                                    result: tool_call.result.clone(),
                                    success: tool_call.success
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
