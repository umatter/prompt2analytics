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
    }
}

/// Render message content based on role
fn render_content(message: &ChatMessage) -> Element {
    if message.role == "user" {
        rsx! {
            p { class: "text-gray-900 dark:text-white whitespace-pre-wrap",
                "{message.content}"
            }
        }
    } else {
        let content = &message.content;
        if message.is_streaming {
            rsx! {
                div { class: "text-gray-800 dark:text-gray-200",
                    {render_markdown(content)}
                    span { class: "inline-block w-2 h-4 ml-0.5 bg-blue-500 animate-pulse" }
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
        // User message - right aligned, blue background
        rsx! {
            div { class: "flex justify-end mb-4 message-enter",
                div { class: "max-w-[80%] rounded-2xl rounded-tr-sm px-4 py-3 bg-blue-600 text-white shadow-sm",
                    // Message content
                    {render_content(message)}

                    // Timestamp
                    div { class: "text-xs text-blue-200 mt-2 text-right",
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
                    // Main message bubble
                    div { class: "rounded-2xl rounded-tl-sm px-4 py-3 bg-gray-100 dark:bg-gray-700 shadow-sm",
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

                    // Tool calls displayed outside the bubble
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
