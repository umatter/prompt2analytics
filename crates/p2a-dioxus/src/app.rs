//! Root application component

use dioxus::prelude::*;

use crate::components::{ChatPanel, ConversationSidebar};
use crate::state::{ChatState, ConversationState, SessionState, Settings};

/// Root application component
#[component]
pub fn App() -> Element {
    // Initialize global state
    let session_state = use_signal(SessionState::new);
    let chat_state = use_signal(ChatState::new);
    let conversation_state = use_signal(ConversationState::new);
    let settings = use_signal(Settings::load);

    // Sidebar visibility state
    let mut sidebar_visible = use_signal(|| true);

    // Provide state to all child components via context
    use_context_provider(|| session_state);
    use_context_provider(|| chat_state);
    use_context_provider(|| conversation_state);
    use_context_provider(|| settings);

    // Initialize session on mount
    use_effect(move || {
        let mut session = session_state;
        spawn(async move {
            if let Err(e) = session.write().initialize().await {
                tracing::error!("Failed to initialize session: {}", e);
            }
        });
    });

    // Handle sidebar toggle
    let handle_toggle_sidebar = move |_| {
        let current = *sidebar_visible.read();
        sidebar_visible.set(!current);
    };

    rsx! {
        div {
            class: "min-h-screen bg-gray-50 dark:bg-gray-900 flex",

            // Sidebar toggle button (visible when sidebar is hidden)
            if !*sidebar_visible.read() {
                button {
                    class: "fixed left-2 top-4 z-50 p-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                    onclick: handle_toggle_sidebar,
                    title: "Show Conversations",
                    svg {
                        class: "w-5 h-5 text-gray-600 dark:text-gray-300",
                        fill: "none",
                        stroke: "currentColor",
                        view_box: "0 0 24 24",
                        path {
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            stroke_width: "2",
                            d: "M4 6h16M4 12h16M4 18h7"
                        }
                    }
                }
            }

            // Conversation sidebar
            if *sidebar_visible.read() {
                div {
                    class: "w-72 flex-shrink-0 relative",
                    // Close sidebar button
                    button {
                        class: "absolute right-2 top-4 z-10 p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors",
                        onclick: handle_toggle_sidebar,
                        title: "Hide Sidebar",
                        svg {
                            class: "w-4 h-4",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M11 19l-7-7 7-7m8 14l-7-7 7-7"
                            }
                        }
                    }
                    ConversationSidebar {}
                }
            }

            // Main chat area
            div {
                class: "flex-1 min-w-0",
                ChatPanel {}
            }
        }
    }
}
