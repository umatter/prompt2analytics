//! MessageList component for displaying all chat messages

use dioxus::prelude::*;

use crate::components::Message;
use crate::state::ChatState;

/// MessageList component - displays all messages with auto-scroll
#[component]
pub fn MessageList() -> Element {
    // Get chat state from context
    let chat_state = use_context::<Signal<ChatState>>();

    // Reference for scrolling using Dioxus's platform-agnostic MountedData
    let mut scroll_element: Signal<Option<std::rc::Rc<MountedData>>> = use_signal(|| None);

    // Auto-scroll to bottom when messages change
    let messages_len = chat_state.read().messages.len();
    use_effect(move || {
        // Depend on messages_len to re-run when messages change
        let _ = messages_len;
        if let Some(ref mounted) = *scroll_element.read() {
            // Use Dioxus's platform-agnostic scroll_to
            let mounted = mounted.clone();
            spawn(async move {
                let _ = mounted.scroll_to(ScrollBehavior::Smooth).await;
            });
        }
    });

    let messages = chat_state.read().messages.clone();

    rsx! {
        div { class: "px-4 py-6",
            if messages.is_empty() {
                // Empty state
                div { class: "flex flex-col items-center justify-center min-h-[400px] text-center",
                    // Icon
                    div { class: "w-16 h-16 mb-4 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center",
                        svg {
                            class: "w-8 h-8 text-gray-400",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "1.5",
                                d: "M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                            }
                        }
                    }
                    h3 { class: "text-lg font-medium text-gray-900 dark:text-white mb-1",
                        "No messages yet"
                    }
                    p { class: "text-sm text-gray-500 dark:text-gray-400 max-w-sm",
                        "Start a conversation to get help with your data analysis. Try asking about loading a dataset or running a regression."
                    }

                    // Quick suggestions
                    div { class: "mt-6 flex flex-wrap gap-2 justify-center",
                        QuickSuggestion { text: "Load a CSV file" }
                        QuickSuggestion { text: "Explain OLS regression" }
                        QuickSuggestion { text: "What tools are available?" }
                    }
                }
            } else {
                // Message list
                div { class: "space-y-2",
                    for message in messages.iter() {
                        Message {
                            key: "{message.id}",
                            message: message.clone()
                        }
                    }

                    // Scroll anchor - use platform-agnostic onmounted
                    div {
                        onmounted: move |event| {
                            scroll_element.set(Some(event.data().clone()));
                        }
                    }
                }
            }
        }
    }
}

/// Quick suggestion chip
#[component]
fn QuickSuggestion(text: String) -> Element {
    let _chat_state = use_context::<Signal<ChatState>>();
    let text_clone = text.clone();

    rsx! {
        button {
            class: "px-3 py-1.5 text-sm text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors",
            onclick: move |_| {
                // This would ideally trigger sending the message
                // For now, just log it
                tracing::info!("Quick suggestion clicked: {}", text_clone);
            },
            "{text}"
        }
    }
}
