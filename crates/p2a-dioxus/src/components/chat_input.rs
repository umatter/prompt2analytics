//! Chat input component with keyboard handling

use dioxus::prelude::*;

use crate::state::ChatState;

/// ChatInput component - textarea with send button
#[component]
pub fn ChatInput(on_send: EventHandler<String>) -> Element {
    let mut chat_state = use_context::<Signal<ChatState>>();
    let mut textarea_value = use_signal(String::new);

    let is_processing = chat_state.read().is_processing;

    // Debug log on every render
    tracing::debug!("[ChatInput] Render, is_processing: {}", is_processing);

    // Handle key events
    let handle_keydown = move |evt: Event<KeyboardData>| {
        let key = evt.key();

        match key {
            Key::Enter => {
                // Enter without shift sends the message
                if !evt.modifiers().shift() {
                    evt.prevent_default();
                    let value = textarea_value.read().clone();
                    if !value.trim().is_empty() && !is_processing {
                        on_send.call(value.clone());
                        textarea_value.set(String::new());
                        chat_state.write().reset_history_index();
                    }
                }
            }
            Key::ArrowUp => {
                // Arrow up navigates history
                let is_empty = textarea_value.read().is_empty();
                if is_empty {
                    evt.prevent_default();
                    chat_state.write().navigate_history_up();
                    let input = chat_state.read().input.clone();
                    textarea_value.set(input);
                }
            }
            Key::ArrowDown => {
                // Arrow down navigates history
                evt.prevent_default();
                chat_state.write().navigate_history_down();
                let input = chat_state.read().input.clone();
                textarea_value.set(input);
            }
            _ => {}
        }
    };

    // Handle input change
    let handle_input = move |evt: Event<FormData>| {
        textarea_value.set(evt.value().clone());
    };

    // Handle send button click
    let handle_send = move |_| {
        tracing::debug!("[ChatInput] Send clicked, is_processing: {}", is_processing);
        let value = textarea_value.read().clone();
        tracing::debug!("[ChatInput] Value: '{}', empty: {}", value, value.trim().is_empty());
        if !value.trim().is_empty() && !is_processing {
            tracing::debug!("[ChatInput] Calling on_send");
            on_send.call(value.clone());
            textarea_value.set(String::new());
            chat_state.write().reset_history_index();
        } else {
            tracing::debug!("[ChatInput] Skipped send - empty or processing");
        }
    };

    rsx! {
        div { class: "px-4 py-4",
            div { class: "flex gap-3 items-end",
                // Textarea container with proper styling
                div { class: "flex-1 relative",
                    textarea {
                        class: "w-full px-4 py-3 pr-12 text-sm bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400 dark:placeholder-gray-500 text-gray-900 dark:text-white transition-all",
                        placeholder: "Type a message...",
                        value: "{textarea_value}",
                        disabled: is_processing,
                        oninput: handle_input,
                        onkeydown: handle_keydown,
                        rows: "1"
                    }
                }

                // Send button
                button {
                    class: if is_processing {
                        "px-4 py-3 bg-gray-400 text-white rounded-xl cursor-not-allowed flex items-center gap-2"
                    } else {
                        "px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-colors flex items-center gap-2 shadow-sm hover:shadow-md"
                    },
                    onclick: handle_send,
                    disabled: is_processing,

                    if is_processing {
                        // Loading spinner
                        svg {
                            class: "w-5 h-5 animate-spin",
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
                        "Processing..."
                    } else {
                        // Send icon
                        svg {
                            class: "w-5 h-5",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                            }
                        }
                        "Send"
                    }
                }
            }

            // Hint text
            div { class: "flex items-center gap-4 mt-2 text-xs text-gray-400 dark:text-gray-500",
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-gray-500 dark:text-gray-400 font-mono", "Enter" }
                    "to send"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-gray-500 dark:text-gray-400 font-mono", "Shift" }
                    "+"
                    kbd { class: "px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-gray-500 dark:text-gray-400 font-mono", "Enter" }
                    "new line"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-gray-500 dark:text-gray-400 font-mono", "\u{2191}/\u{2193}" }
                    "history"
                }
            }
        }
    }
}
