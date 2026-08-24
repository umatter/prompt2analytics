//! MessageList component for displaying all chat messages

use dioxus::prelude::*;

use crate::components::{Message, P2aWordmark};
use crate::state::ChatState;

/// Props for MessageList
#[derive(Props, Clone, PartialEq)]
pub struct MessageListProps {
    /// Callback when a quick suggestion is clicked
    #[props(default)]
    pub on_suggestion: Option<EventHandler<String>>,
}

/// MessageList component - displays all messages with auto-scroll
#[component]
pub fn MessageList(props: MessageListProps) -> Element {
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
        div { class: "py-6 h-full",
            if messages.is_empty() {
                // Empty state: centered wordmark + a row of greetings in one size.
                // Previous layout had "Grüezi" as an oversized headline with the
                // other languages demoted below it; it now sits inline with the
                // rest so no one language feels singled out.
                div { class: "flex flex-col items-center justify-center min-h-[400px] h-full text-center mx-auto",
                    // Wordmark
                    div { class: "mb-20",
                        P2aWordmark { width: 280.0 }
                    }
                    // Multilingual welcome — all greetings at equal weight.
                    div { class: "mb-4 flex flex-wrap justify-center gap-x-3 gap-y-1 text-base text-gray-700 dark:text-gray-300 max-w-md",
                        span { "Welcome" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Bienvenue" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Willkommen" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Grüezi" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Benvenuto" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "欢迎" }
                    }
                    p { class: "text-sm text-gray-500 dark:text-gray-400 max-w-sm",
                        "Start a conversation to get help with your data analysis."
                    }

                    // Quick suggestions
                    div { class: "mt-6 flex flex-wrap gap-2 justify-center",
                        QuickSuggestion {
                            text: "Create a sample dataset with x and y",
                            on_click: props.on_suggestion
                        }
                        QuickSuggestion {
                            text: "Explain OLS regression",
                            on_click: props.on_suggestion
                        }
                        QuickSuggestion {
                            text: "What tools are available?",
                            on_click: props.on_suggestion
                        }
                    }
                }
            } else {
                // Message list
                div { class: "space-y-2 px-4",
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

/// Props for QuickSuggestion
#[derive(Props, Clone, PartialEq)]
struct QuickSuggestionProps {
    text: String,
    #[props(default)]
    on_click: Option<EventHandler<String>>,
}

/// Quick suggestion chip
#[component]
fn QuickSuggestion(props: QuickSuggestionProps) -> Element {
    let text = props.text.clone();
    let text_for_click = props.text.clone();

    rsx! {
        button {
            class: "px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-full cursor-pointer shadow-sm transition-all duration-150 hover:border-orange-500 hover:bg-orange-50 hover:text-orange-700 hover:shadow-md dark:hover:bg-orange-900/30 dark:hover:text-orange-300 active:scale-95 active:shadow-none active:bg-orange-100 dark:active:bg-orange-900/50",
            onclick: move |_| {
                if let Some(ref handler) = props.on_click {
                    handler.call(text_for_click.clone());
                }
            },
            "{text}"
        }
    }
}
